import json
from typing import Any, Dict, List, Tuple, Union, Optional
from difflib import SequenceMatcher
import numpy as np
from scipy.optimize import linear_sum_assignment
from functools import lru_cache
import warnings

from sentence_transformers import SentenceTransformer
import torch
EMBEDDINGS_AVAILABLE = True

class SemanticJSONSimilarity:
    def __init__(self, 
                 structure_weight: float = 0.5,
                 value_weight: float = 0.5,
                 array_order_matters: bool = False,
                 number_tolerance: float = 0.01,
                 string_method: str = 'levenshtein',
                 use_semantic_similarity: bool = True,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 semantic_threshold: float = 0.7,
                 key_semantic_weight: float = 0.7,
                 exact_match_weight: float = 0.3):
        """
        Initialize JSON similarity calculator with semantic similarity support.
        
        Args:
            structure_weight: Weight for structural similarity (0-1)
            value_weight: Weight for value similarity (0-1)
            array_order_matters: Whether array order should be considered
            number_tolerance: Relative tolerance for number comparison
            string_method: Method for string comparison
            use_semantic_similarity: Whether to use embedding-based semantic similarity
            embedding_model: Name of the sentence transformer model
            semantic_threshold: Minimum semantic similarity to consider keys as matching
            key_semantic_weight: Weight for semantic similarity vs exact match for keys
            exact_match_weight: Weight for exact key matching
        """
        self.structure_weight = structure_weight
        self.value_weight = value_weight
        self.array_order_matters = array_order_matters
        self.number_tolerance = number_tolerance
        self.string_method = string_method
        self.use_semantic_similarity = use_semantic_similarity and EMBEDDINGS_AVAILABLE
        self.semantic_threshold = semantic_threshold
        self.key_semantic_weight = key_semantic_weight
        self.exact_match_weight = exact_match_weight
        
        # Normalize weights
        total = self.structure_weight + self.value_weight
        if total > 0:
            self.structure_weight /= total
            self.value_weight /= total
        
        # Normalize key weights
        key_total = self.key_semantic_weight + self.exact_match_weight
        if key_total > 0:
            self.key_semantic_weight /= key_total
            self.exact_match_weight /= key_total
        
        # Initialize embedding model if available
        self.embedding_model = None
        if self.use_semantic_similarity:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                # Warm up the model
                self.embedding_model.encode(["test"], show_progress_bar=False)
            except Exception as e:
                warnings.warn(f"Failed to load embedding model: {e}")
                self.use_semantic_similarity = False
        
        # Cache for embeddings
        self._embedding_cache = {}
    
    @lru_cache(maxsize=1000)
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for a text with caching."""
        if not self.use_semantic_similarity or not self.embedding_model:
            return None
        
        try:
            # Preprocess key names for better semantic understanding
            processed_text = self._preprocess_key_name(text)
            embedding = self.embedding_model.encode(processed_text, show_progress_bar=False)
            return embedding
        except Exception as e:
            warnings.warn(f"Failed to get embedding for '{text}': {e}")
            return None
    
    def _preprocess_key_name(self, key: str) -> str:
        """Preprocess key names for better semantic understanding."""
        # Convert camelCase to words
        import re
        # Add space before capital letters
        processed = re.sub(r'([A-Z])', r' \1', key)
        # Replace underscores and hyphens with spaces
        processed = processed.replace('_', ' ').replace('-', ' ')
        # Convert to lowercase and remove extra spaces
        processed = ' '.join(processed.lower().split())
        return processed
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using embeddings."""
        if not self.use_semantic_similarity:
            return 0.0
        
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def _calculate_key_similarity(self, key1: str, key2: str) -> float:
        """Calculate similarity between two keys using both exact and semantic matching."""
        # Exact match
        if key1 == key2:
            return 1.0
        
        # Calculate exact match similarity (character-based)
        exact_sim = SequenceMatcher(None, key1.lower(), key2.lower()).ratio()
        
        # Calculate semantic similarity
        semantic_sim = self._calculate_semantic_similarity(key1, key2)
        
        # Combine both similarities
        if self.use_semantic_similarity:
            return (self.exact_match_weight * exact_sim + 
                    self.key_semantic_weight * semantic_sim)
        else:
            return exact_sim
    
    def _find_key_mapping(self, keys1: List[str], keys2: List[str]) -> Dict[str, str]:
        """Find optimal mapping between keys using semantic similarity."""
        if not keys1 or not keys2:
            return {}
        
        # Create similarity matrix
        n1, n2 = len(keys1), len(keys2)
        similarity_matrix = np.zeros((n1, n2))
        
        for i, k1 in enumerate(keys1):
            for j, k2 in enumerate(keys2):
                similarity_matrix[i, j] = self._calculate_key_similarity(k1, k2)
        
        # Use Hungarian algorithm for optimal assignment
        # Convert to cost matrix (1 - similarity)
        cost_matrix = 1 - similarity_matrix
        
        # Pad matrix if needed
        if n1 != n2:
            max_n = max(n1, n2)
            padded_cost = np.ones((max_n, max_n))
            padded_cost[:n1, :n2] = cost_matrix
            cost_matrix = padded_cost
        
        # Find optimal assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Create mapping
        mapping = {}
        for i, j in zip(row_indices, col_indices):
            if i < n1 and j < n2:
                similarity = similarity_matrix[i, j]
                if similarity >= self.semantic_threshold:
                    mapping[keys1[i]] = keys2[j]
        
        return mapping
    
    def compare(self, json1: Union[str, dict], json2: Union[str, dict]) -> float:
        """
        Compare two JSON objects and return similarity score (0-1).
        """
        # Parse JSON strings if needed
        if isinstance(json1, str):
            json1 = json.loads(json1)
        if isinstance(json2, str):
            json2 = json.loads(json2)
        
        return self._compare_nodes(json1, json2)
    
    def _compare_nodes(self, node1: Any, node2: Any) -> float:
        """Compare two JSON nodes recursively."""
        # Same reference or both None
        if node1 is node2:
            return 1.0
        
        # Different types
        if type(node1) != type(node2):
            return 0.0
        
        # Handle different types
        if isinstance(node1, dict):
            return self._compare_objects_semantic(node1, node2)
        elif isinstance(node1, list):
            return self._compare_arrays(node1, node2)
        elif isinstance(node1, str):
            return self._compare_strings(node1, node2)
        elif isinstance(node1, (int, float)):
            return self._compare_numbers(node1, node2)
        elif isinstance(node1, bool):
            return 1.0 if node1 == node2 else 0.0
        elif node1 is None:
            return 1.0
        else:
            return 1.0 if node1 == node2 else 0.0
    
    def _compare_objects_semantic(self, obj1: dict, obj2: dict) -> float:
        """Compare two JSON objects using semantic key matching."""
        if not obj1 and not obj2:
            return 1.0
        if not obj1 or not obj2:
            return 0.0
        
        keys1 = list(obj1.keys())
        keys2 = list(obj2.keys())
        
        # Find semantic key mapping
        key_mapping = self._find_key_mapping(keys1, keys2)
        
        # Calculate structural similarity based on mapping
        all_keys = set(keys1) | set(keys2)
        mapped_keys = len(key_mapping)
        
        # Consider exact matches that weren't included in mapping
        exact_matches = set(keys1) & set(keys2)
        for key in exact_matches:
            if key not in key_mapping:
                key_mapping[key] = key
                mapped_keys += 1
        
        structure_sim = mapped_keys / len(all_keys) if all_keys else 1.0
        
        # Calculate value similarity for mapped keys
        if key_mapping:
            value_similarities = []
            similarities_with_weights = []
            
            for k1, k2 in key_mapping.items():
                val_sim = self._compare_nodes(obj1[k1], obj2[k2])
                key_sim = self._calculate_key_similarity(k1, k2)
                
                # Weight value similarity by how similar the keys are
                weighted_sim = val_sim * (0.7 + 0.3 * key_sim)
                value_similarities.append(val_sim)
                similarities_with_weights.append(weighted_sim)
            
            value_sim = sum(similarities_with_weights) / len(similarities_with_weights)
        else:
            value_sim = 0.0
        
        # Combine structural and value similarity
        return (self.structure_weight * structure_sim + 
                self.value_weight * value_sim)
    
    def _compare_arrays(self, arr1: list, arr2: list) -> float:
        """Compare two JSON arrays."""
        if not arr1 and not arr2:
            return 1.0
        if not arr1 or not arr2:
            return 0.0
        
        len1, len2 = len(arr1), len(arr2)
        
        # Structural similarity based on length
        structure_sim = 1 - abs(len1 - len2) / max(len1, len2)
        
        if self.array_order_matters:
            value_sim = self._compare_arrays_ordered(arr1, arr2)
        else:
            value_sim = self._compare_arrays_unordered(arr1, arr2)
        
        return (self.structure_weight * structure_sim + 
                self.value_weight * value_sim)
    
    def _compare_arrays_ordered(self, arr1: list, arr2: list) -> float:
        """Compare arrays considering order."""
        similarities = []
        min_len = min(len(arr1), len(arr2))
        
        for i in range(min_len):
            sim = self._compare_nodes(arr1[i], arr2[i])
            similarities.append(sim)
        
        max_len = max(len(arr1), len(arr2))
        if max_len > min_len:
            similarities.extend([0.0] * (max_len - min_len))
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _compare_arrays_unordered(self, arr1: list, arr2: list) -> float:
        """Compare arrays without considering order using optimal matching."""
        if len(arr1) == 0 or len(arr2) == 0:
            return 0.0
        
        cost_matrix = np.zeros((len(arr1), len(arr2)))
        for i, item1 in enumerate(arr1):
            for j, item2 in enumerate(arr2):
                cost_matrix[i, j] = 1 - self._compare_nodes(item1, item2)
        
        if len(arr1) != len(arr2):
            max_len = max(len(arr1), len(arr2))
            padded_matrix = np.ones((max_len, max_len))
            padded_matrix[:len(arr1), :len(arr2)] = cost_matrix
            cost_matrix = padded_matrix
        
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        total_similarity = 0
        for i, j in zip(row_indices, col_indices):
            if i < len(arr1) and j < len(arr2):
                total_similarity += 1 - cost_matrix[i, j]
        
        return total_similarity / max(len(arr1), len(arr2))
    
    def _compare_strings(self, str1: str, str2: str) -> float:
        """Compare two strings with optional semantic similarity."""
        if str1 == str2:
            return 1.0
        
        if self.string_method == 'exact':
            return 0.0
        elif self.string_method == 'semantic' and self.use_semantic_similarity:
            # Use semantic similarity for string values too
            return self._calculate_semantic_similarity(str1, str2)
        elif self.string_method == 'levenshtein':
            return SequenceMatcher(None, str1, str2).ratio()
        elif self.string_method == 'jaccard':
            set1 = set(str1.lower().split())
            set2 = set(str2.lower().split())
            if not set1 and not set2:
                return 1.0
            intersection = set1 & set2
            union = set1 | set2
            return len(intersection) / len(union) if union else 0.0
        else:
            return SequenceMatcher(None, str1, str2).ratio()
    
    def _compare_numbers(self, num1: float, num2: float) -> float:
        """Compare two numbers with tolerance."""
        if num1 == num2:
            return 1.0
        
        if num1 == 0 or num2 == 0:
            return 0.0 if abs(num1 - num2) > self.number_tolerance else 1.0
        
        rel_diff = abs(num1 - num2) / max(abs(num1), abs(num2))
        
        if rel_diff <= self.number_tolerance:
            return 1.0
        else:
            return max(0, 1 - rel_diff)
    
    def detailed_comparison(self, json1: Union[str, dict], json2: Union[str, dict]) -> Dict[str, Any]:
        """Provide detailed comparison report with semantic analysis."""
        if isinstance(json1, str):
            json1 = json.loads(json1)
        if isinstance(json2, str):
            json2 = json.loads(json2)
        
        report = {
            'overall_similarity': self.compare(json1, json2),
            'type_match': type(json1) == type(json2),
            'semantic_analysis_enabled': self.use_semantic_similarity,
            'details': self._detailed_compare_nodes(json1, json2, path='$')
        }
        
        return report
    
    def _detailed_compare_nodes(self, node1: Any, node2: Any, path: str) -> Dict[str, Any]:
        """Generate detailed comparison for nodes including semantic analysis."""
        result = {
            'path': path,
            'type1': type(node1).__name__,
            'type2': type(node2).__name__,
            'similarity': self._compare_nodes(node1, node2)
        }
        
        if isinstance(node1, dict) and isinstance(node2, dict):
            keys1 = set(node1.keys())
            keys2 = set(node2.keys())
            
            # Find semantic key mapping
            key_mapping = self._find_key_mapping(list(keys1), list(keys2))
            
            result['key_analysis'] = {
                'keys1_only': list(keys1 - set(key_mapping.keys())),
                'keys2_only': list(keys2 - set(key_mapping.values())),
                'exact_matches': list(keys1 & keys2),
                'semantic_matches': [
                    {
                        'key1': k1,
                        'key2': k2,
                        'similarity': self._calculate_key_similarity(k1, k2)
                    }
                    for k1, k2 in key_mapping.items()
                    if k1 != k2
                ]
            }
            
            result['children'] = []
            
            # Analyze mapped keys
            for k1, k2 in key_mapping.items():
                child_path = f"{path}.{k1}"
                if k1 != k2:
                    child_path += f" (→ {k2})"
                child_result = self._detailed_compare_nodes(
                    node1[k1], node2[k2], child_path
                )
                result['children'].append(child_result)
        
        elif isinstance(node1, list) and isinstance(node2, list):
            result['length1'] = len(node1)
            result['length2'] = len(node2)
            result['array_comparison_method'] = 'ordered' if self.array_order_matters else 'unordered'
        
        return result


# Example usage with semantic similarity
if __name__ == "__main__":
    # Example JSONs with semantically similar but different key names
    json1 = {
        "user_name": "John Doe",
        "user_age": 30,
        "email_address": "john.doe@example.com",
        "home_address": {
            "street_name": "123 Main St",
            "city_name": "New York",
            "postal_code": "10001"
        },
        "interests": ["reading", "swimming", "coding"],
        "is_active": True,
        "account_balance": 1500.50
    }
    
    json2 = {
        "name": "John Doe",  # Semantically similar to "user_name"
        "age": 31,  # Semantically similar to "user_age"
        "email": "john.doe@example.com",  # Semantically similar to "email_address"
        "address": {  # Semantically similar to "home_address"
            "street": "123 Main Street",  # Semantically similar to "street_name"
            "city": "New York",  # Semantically similar to "city_name"
            "zip": "10001"  # Semantically similar to "postal_code"
        },
        "hobbies": ["coding", "reading", "running"],  # Semantically similar to "interests"
        "active": True,  # Semantically similar to "is_active"
        "balance": 1499.00  # Semantically similar to "account_balance"
    }
    
    print("=== Semantic JSON Similarity Comparison ===\n")
    
    # Test with semantic similarity enabled
    print("1. With Semantic Similarity:")
    calculator_semantic = SemanticJSONSimilarity(
        structure_weight=0.4,
        value_weight=0.6,
        use_semantic_similarity=True,
        semantic_threshold=0.6,
        string_method='levenshtein'
    )
    
    similarity_semantic = calculator_semantic.compare(json1, json2)
    print(f"   Overall similarity: {similarity_semantic:.2%}")
    
    # Test without semantic similarity
    print("\n2. Without Semantic Similarity (exact key matching only):")
    calculator_exact = SemanticJSONSimilarity(
        structure_weight=0.4,
        value_weight=0.6,
        use_semantic_similarity=False
    )
    
    similarity_exact = calculator_exact.compare(json1, json2)
    print(f"   Overall similarity: {similarity_exact:.2%}")
    
    # Show detailed comparison
    print("\n3. Detailed Semantic Analysis:")
    detailed = calculator_semantic.detailed_comparison(json1, json2)
    
    if 'key_analysis' in detailed['details']:
        analysis = detailed['details']['key_analysis']
        print(f"\n   Exact key matches: {analysis['exact_matches']}")
        print(f"\n   Semantic key matches:")
        for match in analysis['semantic_matches']:
            print(f"      '{match['key1']}' ↔ '{match['key2']}' (similarity: {match['similarity']:.2f})")
    
    # Test with completely different structure
    json3 = {
        "firstName": "John",
        "lastName": "Doe",
        "contact": {
            "email": "john.doe@example.com",
            "phone": "123-456-7890"
        }
    }
    
    print("\n4. Comparing with different structure:")
    similarity_different = calculator_semantic.compare(json1, json3)
    print(f"   Similarity with different structure: {similarity_different:.2%}")