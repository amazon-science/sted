"""
Semantic JSON Structural Consistency Evaluation using Tree Edit Distance

This module enhances the JSON tree consistency evaluation with semantic similarity capabilities.
It combines tree edit distance algorithms with embedding-based semantic similarity to provide
more accurate structural consistency evaluation for JSON outputs.
"""

import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set, Union
from collections import defaultdict
import datetime
from difflib import SequenceMatcher
from functools import lru_cache
import warnings
import re
import time
from bert_score import score as bert_score

from scipy.optimize import linear_sum_assignment
import zss

# Optional imports with proper error handling
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    warnings.warn("LangChain not available. Text splitting will use fallback method.")

try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn("Sentence Transformers not available. Local embedding models will not work.")

try:
    import boto3
    from botocore.exceptions import ClientError
    from botocore.config import Config
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    warnings.warn("Boto3 not available. AWS Bedrock embeddings will not work.")

def getEmbeddings(text, model_id, bedrock_client, max_retries=10, initial_delay=2, output_embedding_length=1024):
    """
    Get embeddings from Bedrock with proper retry and connection handling
    """
    def exponential_delay(attempt):
        # Add jitter to prevent thundering herd
        jitter = 0.1 * initial_delay * np.random.random()
        return initial_delay * (2 ** attempt) + jitter
    
    if model_id == "amazon.titan-embed-text-v1":
        request_body = {
            "inputText": text
        }
    elif model_id == "amazon.titan-embed-text-v2:0":
        request_body = {
            "inputText": text,
            "dimensions": output_embedding_length,
            "normalize": True,
            "embeddingTypes": ["float"]
        }
    elif model_id == "cohere.embed-multilingual-v3":
        request_body = {
            "texts": [text],
            "input_type": "clustering",
            "truncate": "END",
            "dimensions": output_embedding_length,
            "normalize": True,
            "embeddingTypes": ["float"]
        }
    else:
        raise ValueError(f"Unknown model_id: {model_id}")

    for attempt in range(max_retries):
        try:
            body = json.dumps(request_body)
            response = bedrock_client.invoke_model(
                body=body,
                modelId=model_id,
                accept="application/json",
                contentType="application/json")
            response_body = json.loads(response.get("body").read())
            
            # Handle different response formats
            if model_id == "cohere.embed-multilingual-v3":
                embedding = response_body.get("embeddings", [[]])[0]
            else:
                embedding = response_body.get("embedding", [])
                
            return np.array(embedding).astype(np.float32)
        
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            
            # Handle specific errors differently
            if error_code in ['ThrottlingException', 'TooManyRequestsException', 'ServiceUnavailable']:
                # These are definitely retryable
                print(f"Throttling detected on attempt {attempt + 1}: {str(e)}")
            elif error_code in ['InternalServerError', 'ServiceError']:
                # Server-side errors that might resolve
                print(f"Server error on attempt {attempt + 1}: {str(e)}")
            else:
                print(f"Client error on attempt {attempt + 1}: {str(e)}")
            
            if attempt == max_retries - 1:
                print(f"Max retries ({max_retries}) exceeded. Giving up.")
                raise  # If this was the last attempt, re-raise the exception
            
            delay = exponential_delay(attempt)
            print(f"Retrying in {delay:.2f} seconds...")
            time.sleep(delay)
        
        except Exception as e:
            # Catch other exceptions (like connection errors)
            print(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                raise
            
            delay = exponential_delay(attempt)
            print(f"Retrying in {delay:.2f} seconds...")
            time.sleep(delay)

    # This should be unreachable with the retry logic above
    raise Exception("Max retries reached. Unable to get embeddings.")

class JsonNode:
    """Node representation for JSON tree structure."""
    
    def __init__(self, label: str, value: Any = None, node_type: str = None):
        """
        Initialize a JSON node.
        
        Args:
            label: The label (key or index) of the node
            value: The value of the node (for leaf nodes)
            node_type: The type of the node ('object', 'array', or value type)
        """
        self.label = label
        self.value = value
        self.children = []
        self.node_type = node_type or self._determine_type(value)
        self.path = label  # Full path to this node
        
    def _determine_type(self, value: Any) -> str:
        """Determine the type of a node based on its value."""
        if value is None:
            return "null"
        elif isinstance(value, dict):
            return "object"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, (int, float)):
            return "number"
        elif isinstance(value, str):
            return "string"
        else:
            return str(type(value).__name__)
    
    def add_child(self, child: 'JsonNode'):
        """Add a child node."""
        self.children.append(child)
    
    def get_children(self):
        """Get all children of this node (required for zss)."""
        return self.children
    
    def get_label(self):
        """Get the label of this node (required for zss)."""
        return self.label
    
    def __str__(self):
        """String representation of the node."""
        if self.value is not None:
            return f"{self.path} ({self.node_type}): {self.value}"
        return f"{self.path} ({self.node_type})"
    
    def __repr__(self):
        return self.__str__()

class SemanticJsonTreeConsistencyEvaluator:
    """Evaluator for JSON structural consistency using Tree Edit Distance with semantic similarity."""
    
    def __init__(self, 
                 schema_aware: bool = False, 
                 array_order_matters: bool = True,
                 path_weight_decay: float = 0.9,
                 type_change_cost: Dict[Tuple[str, str], float] = None,
                 required_fields: Set[str] = None,
                 use_semantic_similarity: bool = True,
                 model_id: str = 'all-MiniLM-L6-v2',
                 semantic_threshold: float = 0.7,
                 key_semantic_weight: float = 0.7,
                 exact_match_weight: float = 0.3,
                 string_method: str = 'bertscore',  # Changed default to bertscore
                 number_tolerance: float = 0.01,
                 use_hungarian: bool = True,
                 long_string_method: str = 'bertscore',  # Changed default to bertscore
                 use_langchain_splitter: bool = True,
                 chunk_size: int = 300,
                 chunk_overlap: int = 50):
        """
        Initialize the evaluator with semantic capabilities.
        
        Args:
            schema_aware: Whether to use schema information if available
            array_order_matters: Whether the order of array elements matters
            path_weight_decay: Weight decay factor for deeper paths (0-1)
            type_change_cost: Custom costs for type changes
            required_fields: Set of required field paths
            use_semantic_similarity: Whether to use embedding-based semantic similarity
            model_id: Name of the sentence transformer model or Bedrock model ID
            semantic_threshold: Minimum semantic similarity to consider keys as matching
            key_semantic_weight: Weight for semantic similarity vs exact match for keys
            exact_match_weight: Weight for exact key matching
            string_method: Method for string comparison ('levenshtein', 'semantic', 'exact', 'jaccard', 'bertscore')
            number_tolerance: Relative tolerance for number comparison
            use_hungarian: Whether to use Hungarian algorithm for array matching and long string comparison
            long_string_method: Method for long string comparison ('hungarian', 'bertscore', 'cosine', 'direct')
            use_langchain_splitter: Whether to use LangChain for text splitting
            chunk_size: Size of chunks for text splitting
            chunk_overlap: Overlap between chunks
        """
        self.schema_aware = schema_aware
        self.array_order_matters = array_order_matters
        self.path_weight_decay = path_weight_decay
        self.type_change_cost = type_change_cost or self._default_type_change_costs()
        self.required_fields = required_fields or set()
        
        # Semantic similarity parameters
        self.use_semantic_similarity = use_semantic_similarity
        self.semantic_threshold = semantic_threshold
        self.key_semantic_weight = key_semantic_weight
        self.exact_match_weight = exact_match_weight
        self.string_method = string_method
        self.number_tolerance = number_tolerance
        
        # Hungarian algorithm parameters
        self.use_hungarian = use_hungarian
        self.long_string_method = long_string_method
        
        # Import BERTScore if needed
        self.bert_score_available = False
        if self.long_string_method == 'bertscore' or self.string_method == 'bertscore':
            try:
                self.bert_score = bert_score
                self.bert_score_available = True
            except ImportError:
                warnings.warn("BERTScore not available. Falling back to alternative methods.")
                if self.long_string_method == 'bertscore':
                    self.long_string_method = 'cosine'
                if self.string_method == 'bertscore':
                    self.string_method = 'levenshtein'
        
        # Normalize key weights
        key_total = self.key_semantic_weight + self.exact_match_weight
        if key_total > 0:
            self.key_semantic_weight /= key_total
            self.exact_match_weight /= key_total
        
        # Initialize embedding model if available
        self.embedding_model = None
        self.bedrock_client = None
        self.model_id = model_id
        
        if self.use_semantic_similarity:
            if self.model_id in ["amazon.titan-embed-text-v1", "amazon.titan-embed-text-v2:0", "cohere.embed-multilingual-v3"]:
                if not BOTO3_AVAILABLE:
                    warnings.warn("Boto3 not available. Cannot use Bedrock embeddings.")
                    self.use_semantic_similarity = False
                else:
                    bedrock_config = Config(
                        max_pool_connections=50,  # Increase connection pool size
                        retries={'max_attempts': 5, 'mode': 'adaptive'},
                        connect_timeout=10,
                        read_timeout=30,
                        tcp_keepalive=True
                    )
                    session = boto3.Session()
                    
                    self.bedrock_client = session.client(
                        'bedrock-runtime',
                        config=bedrock_config
                    )
            else:
                if not SENTENCE_TRANSFORMERS_AVAILABLE:
                    warnings.warn("Sentence Transformers not available. Cannot use local embedding models.")
                    self.use_semantic_similarity = False
                else:
                    try:
                        self.embedding_model = SentenceTransformer(self.model_id)
                        # Warm up the model
                        self.embedding_model.encode(["test"], show_progress_bar=False)
                    except Exception as e:
                        warnings.warn(f"Failed to load embedding model: {e}")
                        self.use_semantic_similarity = False
        
        # Cache for embeddings
        self._embedding_cache = {}
        
        # Text splitting configuration
        self.use_langchain_splitter = use_langchain_splitter and LANGCHAIN_AVAILABLE
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def _default_type_change_costs(self) -> Dict[Tuple[str, str], float]:
        """Define default costs for type changes."""
        costs = {}
        types = ["object", "array", "string", "number", "boolean", "null"]
        
        # Default cost is 1.0
        for t1 in types:
            for t2 in types:
                costs[(t1, t2)] = 1.0
        
        # Same type has zero cost
        for t in types:
            costs[(t, t)] = 0.0
        
        # Lower costs for some type conversions
        costs[("string", "number")] = costs[("number", "string")] = 0.5
        costs[("boolean", "string")] = costs[("string", "boolean")] = 0.7
        costs[("number", "boolean")] = costs[("boolean", "number")] = 0.7
        costs[("null", "string")] = costs[("string", "null")] = 0.5
        costs[("null", "number")] = costs[("number", "null")] = 0.5
        
        # Higher costs for structure changes
        costs[("object", "array")] = costs[("array", "object")] = 1.5
        costs[("object", "string")] = costs[("string", "object")] = 1.5
        costs[("object", "number")] = costs[("number", "object")] = 1.5
        costs[("array", "string")] = costs[("string", "array")] = 1.5
        costs[("array", "number")] = costs[("number", "array")] = 1.5
        
        return costs
    
    @lru_cache(maxsize=1000)
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for a text with caching."""
        if not self.use_semantic_similarity:
            return None
        
        if self.bedrock_client:
            try:
                return getEmbeddings(text, self.model_id, self.bedrock_client)
            except Exception as e:
                warnings.warn(f"Failed to get Bedrock embedding for '{text}': {e}")
                return None
        elif self.embedding_model:
            try:
                # Preprocess key names for better semantic understanding
                processed_text = self._preprocess_key_name(text)
                embedding = self.embedding_model.encode(processed_text, show_progress_bar=False)
                return embedding
            except Exception as e:
                warnings.warn(f"Failed to get embedding for '{text}': {e}")
                return None
        
        return None
    
    def _preprocess_key_name(self, key: str) -> str:
        """Preprocess key names for better semantic understanding."""
        if not key:
            return ""
        
        # Handle acronyms (e.g., "HTTPSConnection" -> "HTTPS Connection")
        processed = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', key)
        processed = re.sub(r'([a-z\d])([A-Z])', r'\1 \2', processed)
        
        # Replace separators
        processed = re.sub(r'[_\-\.]+', ' ', processed)
        
        # Clean up and lowercase
        return ' '.join(processed.lower().split())
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using embeddings."""
        if not self.use_semantic_similarity:
            return 0.0
        
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Ensure embeddings are 1D arrays
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        similarity = dot_product / (norm1 * norm2)
        return float(np.clip(similarity, -1.0, 1.0))
    
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
    
    def json_to_tree(self, json_obj: Any, path: str = "", parent_path: str = "") -> JsonNode:
        """
        Convert a JSON object to a tree representation.
        
        Args:
            json_obj: The JSON object to convert
            path: The current path in the JSON structure
            parent_path: The path of the parent node
            
        Returns:
            A JsonNode representing the root of the tree
        """
        full_path = path if path else "root"
        
        if isinstance(json_obj, dict):
            # Create a node for the object
            node = JsonNode(full_path, node_type="object")
            
            # Add children for each key-value pair
            for key, value in sorted(json_obj.items()):  # Sort for deterministic behavior
                child_path = f"{full_path}.{key}" if full_path != "root" else key
                child = self.json_to_tree(value, child_path, full_path)
                node.add_child(child)
        
        elif isinstance(json_obj, list):
            # Create a node for the array
            """
            node = JsonNode(full_path, node_type="array")
            
            if len(json_obj) > 0 and not isinstance(json_obj[0], str):
                # Add children for each array item
                for i, item in enumerate(json_obj):
                    child_path = f"{full_path}[{i}]" if full_path != "root" else f"[{i}]"
                    child = self.json_to_tree(item, child_path, full_path)
                    node.add_child(child)
            """
            if len(json_obj) > 0:
                if isinstance(json_obj[0], dict):
                    json_obj = [self.json_to_tree(obj, path="") for obj in json_obj]
                    
                node = JsonNode(full_path, json_obj)
        else:
            # Create a leaf node for primitive values
            node = JsonNode(full_path, json_obj)
        
        return node

    def calculate_path_weight(self, path: str) -> float:
        """
        Calculate weight for a path based on its depth.
        
        Args:
            path: The path to calculate weight for
            
        Returns:
            A weight factor between 0 and 1
        """
        # Count the number of path segments
        if path == "root":
            depth = 0
        else:
            # Count dots and array indices
            depth = path.count('.') + path.count('[')
        
        # Apply exponential decay based on depth
        weight = self.path_weight_decay ** depth
        
        # Increase weight for required fields
        if path in self.required_fields:
            weight *= 1.5
        
        return weight
    
    def insert_cost(self, node: JsonNode) -> float:
        """
        Calculate the cost of inserting a node.
        
        Args:
            node: The node to insert
            
        Returns:
            The cost of insertion
        """
        # Base cost is 1.0
        cost = 1.0
        
        # Apply path-based weighting
        cost *= self.calculate_path_weight(node.path)
        
        return cost
    
    def delete_cost(self, node: JsonNode) -> float:
        """
        Calculate the cost of deleting a node.
        
        Args:
            node: The node to delete
            
        Returns:
            The cost of deletion
        """
        # Base cost is 1.0
        cost = 1.0
        
        # Apply path-based weighting
        cost *= self.calculate_path_weight(node.path)
        
        # Higher cost for deleting required fields
        if node.path in self.required_fields:
            cost *= 2.0
        
        return cost
    
    def _normalize_path(self, path: str) -> str:
        """
        Normalize a path by replacing array indices with a placeholder.
        This allows structural comparison while ignoring specific array indices.
        
        Args:
            path: The path to normalize
            
        Returns:
            Normalized path
        """
        # Replace array indices like [0], [1], etc. with [*]
        normalized = re.sub(r'\[\d+\]', '[*]', path)
        return normalized
    
    def _calculate_node_similarity(self, node1: JsonNode, node2: JsonNode) -> Dict[str, Any]:
        """
        Calculate similarity metrics between two nodes.
        This shared method is used by both are_nodes_equal and update_cost.
        
        Args:
            node1: First node
            node2: Second node
            
        Returns:
            Dictionary with similarity metrics
        """
        # Extract key names for comparison
        key1 = node1.label.split('.')[-1] if '.' in node1.label else node1.label
        key2 = node2.label.split('.')[-1] if '.' in node2.label else node2.label
        
        # Remove array brackets for comparison
        key1 = re.sub(r'\[\d+\]', '', key1)
        key2 = re.sub(r'\[\d+\]', '', key2)
        
        # Calculate basic metrics
        type_match = node1.node_type == node2.node_type
        
        # Compare paths
        path1 = self._normalize_path(node1.path)
        path2 = self._normalize_path(node2.path)
        path_match = path1 == path2
        
        # Check if both are array indices
        both_array_indices = '[' in node1.label and '[' in node2.label
        
        # Calculate key similarity
        key_sim = self._calculate_key_similarity(key1, key2) if not both_array_indices else 1.0
        
        # Calculate value similarity for leaf nodes
        value_sim = 0.0
        if not node1.children and not node2.children and type_match:
            if node1.node_type == "string":
                value_sim = self._compare_strings(str(node1.value), str(node2.value))
            elif node1.node_type == "number":
                value_sim = self._compare_numbers(float(node1.value), float(node2.value))
            elif node1.node_type == "array":
                value_sim = self._compare_arrays_unordered(list(node1.value), list(node2.value))
            elif node1.value == node2.value:  # For boolean, null, etc.
                value_sim = 1.0
        
        return {
            "type_match": type_match,
            "path_match": path_match,
            "both_array_indices": both_array_indices,
            "key_sim": key_sim,
            "value_sim": value_sim,
            "key1": key1,
            "key2": key2,
            "is_leaf": not node1.children and not node2.children
        }
    
    def _compare_strings(self, str1: str, str2: str) -> float:
        """Compare two strings with optional semantic similarity and chunking for long text."""
        # Quick equality check
        if str1 == str2:
            return 1.0
        
        # For BERTScore, bypass the length check since it handles all lengths well
        if self.string_method == 'bertscore' and self.bert_score_available:
            try:
                print(f"Running bertscore!!!!")
                P, R, F1 = self.bert_score([str1], [str2], lang="en")
                return float(F1.item())
            except Exception as e:
                warnings.warn(f"Error using BERTScore: {e}. Falling back to length-based processing.")
                # Fall back to length-based processing below
        
        # For very short strings or exact matching mode
        if self.string_method == 'exact':
            return 1.0 if str1 == str2 else 0.0
        
        # For longer text, use chunking approach
        if len(str1) > 100 or len(str2) > 100:
            return self._compare_long_strings(str1, str2)
        
        # For shorter strings, use the basic comparison method
        return self._compare_strings_simple(str1, str2)
    
    def _compare_long_strings(self, str1: str, str2: str) -> float:
        """Compare long strings using the specified method.
        
        Available methods:
        - 'hungarian': Break into chunks and use Hungarian algorithm for optimal matching
        - 'bertscore': Use BERTScore for semantic similarity
        - 'cosine': Use cosine similarity between embeddings
        - 'direct': Use direct string comparison without chunking
        """
        # For moderately sized text, consider direct comparison first
        if len(str1) < 200 and len(str2) < 200:
            # For shorter texts, direct comparison might be more accurate
            direct_sim = self._compare_strings_simple(str1, str2)
            # If direct similarity is high, just use it
            if direct_sim > 0.8:
                return direct_sim
        
        # Use the specified method for long string comparison
        if self.long_string_method == 'direct':
            return self._compare_strings_simple(str1, str2)
        
        elif self.long_string_method == 'bertscore' and self.bert_score_available:
            # Use BERTScore for semantic similarity
            try:
                P, R, F1 = self.bert_score([str1], [str2], lang="en")
                return float(F1.item())
            except Exception as e:
                warnings.warn(f"Error using BERTScore: {e}. Falling back to cosine similarity.")
                # Fall back to cosine similarity
                return self._compare_long_strings_cosine(str1, str2)
        
        elif self.long_string_method == 'cosine' or (self.long_string_method == 'bertscore' and not self.bert_score_available):
            # Use cosine similarity between embeddings
            return self._compare_long_strings_cosine(str1, str2)
        
        else:  # Default to Hungarian algorithm
            return self._compare_long_strings_hungarian(str1, str2)
    
    def _compare_long_strings_cosine(self, str1: str, str2: str) -> float:
        """Compare long strings using cosine similarity between embeddings."""
        if not self.use_semantic_similarity:
            return self._compare_strings_simple(str1, str2)
        
        # Get embeddings for the entire strings
        emb1 = self._get_embedding(str1)
        emb2 = self._get_embedding(str2)
        
        if emb1 is None or emb2 is None:
            return self._compare_strings_simple(str1, str2)
        
        # Ensure embeddings are 1D arrays
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()
        
        # Calculate cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return self._compare_strings_simple(str1, str2)
            
        similarity = dot_product / (norm1 * norm2)
        return float(np.clip(similarity, -1.0, 1.0))
    
    def _compare_long_strings_hungarian(self, str1: str, str2: str) -> float:
        """Compare long strings by breaking them into chunks and using Hungarian algorithm for optimal matching."""
        # Split into chunks (sentences or paragraphs)
        chunks1 = self._split_into_chunks(str1)
        chunks2 = self._split_into_chunks(str2)
        
        # If chunking failed or produced very few chunks, fall back to basic comparison
        if len(chunks1) <= 1 or len(chunks2) <= 1:
            return self._compare_strings_simple(str1, str2)
        
        # Create similarity matrix between all chunks
        similarity_matrix = np.zeros((len(chunks1), len(chunks2)))
        
        for i, chunk1 in enumerate(chunks1):
            for j, chunk2 in enumerate(chunks2):
                # Use appropriate method for chunk comparison
                similarity_matrix[i, j] = self._compare_strings_simple(chunk1, chunk2)
        
        if self.use_hungarian:
            # Use Hungarian algorithm to find optimal matching between chunks
            row_ind, col_ind = linear_sum_assignment(-similarity_matrix)  # Negate for max similarity
            
            # Calculate total similarity of matched chunks
            matched_similarities = [similarity_matrix[i, j] for i, j in zip(row_ind, col_ind)]
            
            # Calculate normalized similarity (accounting for unmatched chunks)
            # This divides by the max number of chunks to properly penalize unmatched content
            total_similarity = sum(matched_similarities)
            max_chunks = max(len(chunks1), len(chunks2))
            normalized_similarity = total_similarity / max_chunks if max_chunks > 0 else 0.0
            
            # Also calculate traditional average for comparison
            avg_similarity = sum(matched_similarities) / len(matched_similarities) if matched_similarities else 0.0
            
            # Calculate coverage (what percentage of chunks are matched well)
            good_matches = sum(1 for sim in matched_similarities if sim > 0.7)
            coverage = good_matches / max_chunks if max_chunks > 0 else 0.0
            
            # Combine normalized similarity and coverage
            # Weight similarity more heavily but ensure coverage affects the score
            chunk_sim = 0.8 * normalized_similarity + 0.2 * coverage
            
            # For debugging purposes, store these metrics
            self._last_hungarian_metrics = {
                "avg_similarity": avg_similarity,
                "normalized_similarity": normalized_similarity,
                "coverage": coverage,
                "matched_count": len(matched_similarities),
                "chunks1_count": len(chunks1),
                "chunks2_count": len(chunks2)
            }
        else:
            # Without Hungarian algorithm, use average of best matches for each chunk in chunks1
            best_matches = []
            for i in range(len(chunks1)):
                if len(chunks2) > 0:
                    best_match = max(similarity_matrix[i, :])  # Best match for this chunk
                    best_matches.append(best_match)
            
            # Calculate average similarity
            chunk_sim = sum(best_matches) / len(best_matches) if best_matches else 0.0
        
        # For better results, compare with direct similarity and take the higher value
        # This handles cases where chunking might not be beneficial
        direct_sim = self._compare_strings_simple(str1, str2)
        
        # Return the better of the two approaches
        return max(chunk_sim, direct_sim * 0.9)  # Slight penalty for direct comparison
    
    def _compare_strings_simple(self, str1: str, str2: str) -> float:
        """Basic string comparison without chunking."""
        if str1 == str2:
            return 1.0
            
        if self.string_method == 'exact':
            return 1.0 if str1 == str2 else 0.0
        elif self.string_method == 'bertscore' and self.bert_score_available:
            # Use BERTScore for free text comparison
            try:
                P, R, F1 = self.bert_score([str1], [str2], lang="en")
                return float(F1.item())
            except Exception as e:
                warnings.warn(f"Error using BERTScore for string comparison: {e}. Falling back to levenshtein.")
                return SequenceMatcher(None, str1, str2).ratio()
        elif self.string_method == 'semantic' and self.use_semantic_similarity:
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
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into meaningful chunks based on content type."""
        # Check if this is code or structured data
        if self._is_structured_text(text):
            return self._split_structured_text(text)
        else:
            return self._split_natural_text(text)
    
    def _is_structured_text(self, text: str) -> bool:
        """Detect if text is code, JSON, or other structured format."""
        # Check for code indicators
        code_indicators = [
            text.count('{') > 3 and text.count('}') > 3,  # Likely code blocks
            text.count('[') > 3 and text.count(']') > 3,  # Likely arrays
            text.strip().startswith(('def ', 'function', 'class ', '<?php', '<html', '#!/')),
            text.count('import ') > 1 or text.count('from ') > 1,  # Python imports
            text.count(';') > text.count('.') * 2,  # Likely code with semicolons
            text.count('=') > text.count(' ') / 10  # Many assignments
        ]
        
        return any(code_indicators)
    
    def _split_structured_text(self, text: str) -> List[str]:
        """Split structured text (code, JSON) into logical chunks."""
        # Split by lines first
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # For very short texts, return as is
        if len(lines) <= 5:
            return lines
        
        # Group lines into logical blocks (e.g., functions, blocks)
        chunks = []
        current_chunk = []
        indent_level = 0
        
        for line in lines:
            # Check for block start/end
            if '{' in line:
                indent_level += line.count('{')
            if '}' in line:
                indent_level -= line.count('}')
            
            current_chunk.append(line)
            
            # End of logical block or function
            if (indent_level == 0 and line.endswith((';', '}')) or 
                line.startswith(('def ', 'class ', 'function')) or
                len(current_chunk) > 10):  # Avoid chunks getting too large
                
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
        
        # Add any remaining lines
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _split_natural_text(self, text: str) -> List[str]:
        """Split natural language text into sentences or paragraphs using LangChain if available and enabled."""
        # For short text, don't split at all
        if len(text) < 100:
            return [text]
        
        # Try to use LangChain's text splitters if available and enabled
        if self.use_langchain_splitter and LANGCHAIN_AVAILABLE:
            try:
                # Create a text splitter that tries to create semantically meaningful chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]  # Try these separators in order
                )
                
                # Split the text
                chunks = text_splitter.split_text(text)
                
                # If we got reasonable chunks, return them
                if len(chunks) >= 2:
                    return chunks
            except Exception as e:
                warnings.warn(f"Error using LangChain splitter: {e}")
        
        # Custom splitting logic (fallback)
        import re
        
        # Try to split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If we get very few sentences, try to split by paragraphs
        if len(sentences) <= 2:
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            if len(paragraphs) > len(sentences):
                return paragraphs
        
        # If still too few chunks and text is long enough, try other splitting methods
        if len(sentences) <= 2 and len(text) > 200:
            # Try splitting by other punctuation
            chunks = re.split(r'(?<=[,;:])\s+', text)
            chunks = [c.strip() for c in chunks if c.strip()]
            
            # If still too few, use fixed-length chunks with overlap
            if len(chunks) <= 2:
                chunk_size = min(self.chunk_size, len(text) // 3)
                overlap = min(self.chunk_overlap, chunk_size // 2)
                chunks = []
                for i in range(0, len(text), chunk_size - overlap):
                    chunks.append(text[i:i+chunk_size])
                return chunks
            
            return chunks
        
        # If we have a reasonable number of sentences, use those
        if len(sentences) > 1:
            return sentences
        
        # Fall back to the original text if all splitting attempts failed
        return [text]
    
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
            
    def update_cost(self, node1: JsonNode, node2: JsonNode) -> float:
        """
        Calculate the cost of updating a node.
        
        Args:
            node1: The source node
            node2: The target node
            
        Returns:
            The cost of update
        """        
        # Special handling for array nodes when Hungarian algorithm is enabled
        if (node1.node_type == "array" and node2.node_type == "array" and 
            self.use_hungarian and not self.array_order_matters):
            # Use Hungarian algorithm for array matching
            print(f"node1: {node1}, {node1.value}")
            similarity = self._compare_arrays_unordered(list(node1.value), list(node2.value))
            
            print(f"array sim: {similarity}")
            # Convert similarity to cost
            cost = 1.0 - similarity
            # Apply path-based weighting
            path_weight1 = self.calculate_path_weight(node1.path)
            path_weight2 = self.calculate_path_weight(node2.path)
            avg_path_weight = (path_weight1 + path_weight2) / 2.0
            cost *= avg_path_weight
            return cost
            
        # Get similarity metrics from shared calculation
        sim = self._calculate_node_similarity(node1, node2)
        
        # Base cost for type change
        type_cost = self.type_change_cost.get(
            (node1.node_type, node2.node_type), 
            1.0  # Default cost if not specified
        )
        
        # Calculate value cost
        if sim["type_match"] and sim["is_leaf"]:
            # For leaf nodes with same type, use the value similarity
            value_cost = 1.0 - sim["value_sim"]
        else:
            value_cost = 0.0  # No value cost if types are different or not leaf nodes
        
        # Reduce cost based on key similarity
        key_factor = 1.0 - (sim["key_sim"] * 0.5)  # At most 50% reduction based on key similarity
        
        # Combine costs
        cost = (type_cost + value_cost * 0.5) * key_factor  # Weight value cost less than type cost
        
        # Apply path-based weighting - consider both paths
        # Use the average of both path weights for symmetry
        path_weight1 = self.calculate_path_weight(node1.path)
        path_weight2 = self.calculate_path_weight(node2.path)
        avg_path_weight = (path_weight1 + path_weight2) / 2.0
        
        cost *= avg_path_weight
        
        return cost
    
    def _compare_arrays_unordered(self, arr1: List[JsonNode], arr2: List[JsonNode]) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Compare arrays without considering order using optimal matching.
        
        Args:
            arr1: First array of nodes
            arr2: Second array of nodes
            
        Returns:
            Tuple of (similarity_score, matching_pairs)
        """
        if len(arr1) == 0 or len(arr2) == 0:
            return 0.0, []
        
        # Create cost matrix
        cost_matrix = np.zeros((len(arr1), len(arr2)))
        for i, item1 in enumerate(arr1):
            for j, item2 in enumerate(arr2):
                # Cost is inverse of similarity
                if isinstance(item1, str) and isinstance(item2, str):
                    cost_matrix[i, j] = self._compare_strings_simple(item1, item2)
                elif isinstance(item1, dict) and isinstance(item2, dict):
                    cost_matrix[i, j] = self._calculate_node_similarity(item1, item2)
                elif isinstance(item1, dict) and isinstance(item2, dict):
                    cost_matrix[i, j] = self._compare_numbers(item1, item2)
                else:
                    cost_matrix[i, j] = 0 # similarity is 0 if arr1 and arr2 have different type
        
        # Normalize cost matrix to [0, 1] range
        max_cost = np.max(cost_matrix) if np.max(cost_matrix) > 0 else 1.0
        cost_matrix = cost_matrix / max_cost
        
        if self.use_hungarian:
            # Pad matrix if needed
            if len(arr1) != len(arr2):
                max_len = max(len(arr1), len(arr2))
                padded_matrix = np.ones((max_len, max_len))
                padded_matrix[:len(arr1), :len(arr2)] = cost_matrix
                cost_matrix = padded_matrix
            
            # Find optimal assignment using Hungarian algorithm
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Calculate similarity and collect matching pairs
            total_similarity = 0
            matching_pairs = []
            
            for i, j in zip(row_indices, col_indices):
                if i < len(arr1) and j < len(arr2):
                    similarity = 1 - cost_matrix[i, j]
                    total_similarity += similarity
                    matching_pairs.append((i, j))
            
            avg_similarity = total_similarity / max(len(arr1), len(arr2))
        else:
            # Without Hungarian algorithm, use greedy matching
            # For each element in arr1, find the best match in arr2
            total_similarity = 0
            matching_pairs = []
            used_indices = set()
            
            # Sort by similarity (lowest cost first)
            pairs = []
            for i in range(len(arr1)):
                for j in range(len(arr2)):
                    pairs.append((i, j, cost_matrix[i, j]))
            
            pairs.sort(key=lambda x: x[2])  # Sort by cost (ascending)
            
            # Greedy assignment
            for i, j, cost in pairs:
                if i not in [p[0] for p in matching_pairs] and j not in [p[1] for p in matching_pairs]:
                    similarity = 1 - cost
                    total_similarity += similarity
                    matching_pairs.append((i, j))
                    
                    # Stop when we've matched all elements in either array
                    if len(matching_pairs) == min(len(arr1), len(arr2)):
                        break
            
            avg_similarity = total_similarity / max(len(arr1), len(arr2))
        
        return avg_similarity
    
    def calculate_tree_edit_distance(self, json1: Dict[str, Any], json2: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Calculate tree edit distance between two JSON objects.
        
        Args:
            json1: First JSON object
            json2: Second JSON object
            
        Returns:
            Tuple of (similarity_score, edit_operations)
        """
        # Convert JSONs to trees
        tree1 = self.json_to_tree(json1)
        tree2 = self.json_to_tree(json2)
        
        # Use Zhang-Shasha algorithm from zss
        # Check which version of zss API we're using
        try:
            # Newer zss versions
            distance = zss.distance(
                tree1, tree2,
                get_children=lambda x: x.get_children(),
                insert_cost=self.insert_cost,
                remove_cost=self.delete_cost,
                update_cost=self.update_cost
            )
            
            # Get edit operations (if available in zss)
            try:
                ops = zss.operations(
                    tree1, tree2,
                    get_children=lambda x: x.get_children(),
                    insert_cost=self.insert_cost,
                    remove_cost=self.delete_cost,
                    update_cost=self.update_cost
                )
                operations = self._format_operations(ops, tree1, tree2)
            except AttributeError:
                # If operations not available in zss
                operations = []
        except TypeError:
            # Older zss versions that require get_label
            distance = zss.distance(
                tree1, tree2,
                get_children=lambda x: x.get_children(),
                #get_label=lambda x: x.get_label(),
                insert_cost=self.insert_cost,
                remove_cost=self.delete_cost,
                update_cost=self.update_cost
            )
            
            # Get edit operations (if available in zss)
            try:
                ops = zss.operations(
                    tree1, tree2,
                    get_children=lambda x: x.get_children(),
                    get_label=lambda x: x.get_label(),
                    insert_cost=self.insert_cost,
                    remove_cost=self.delete_cost,
                    update_cost=self.update_cost
                )
                operations = self._format_operations(ops, tree1, tree2)
            except AttributeError:
                # If operations not available in zss
                operations = []
        
        # Calculate tree sizes for normalization
        size1 = self._count_nodes(tree1)
        size2 = self._count_nodes(tree2)
        max_size = max(size1, size2)
        
        # Normalize distance to [0, 1] range
        if max_size > 0:
            normalized_distance = distance / max_size
        else:
            normalized_distance = 0.0
        
        # Convert to similarity score (1 - normalized distance)
        similarity = 1.0 - normalized_distance
        
        return similarity, operations
        
    def _count_nodes(self, node: JsonNode) -> int:
        """Count the number of nodes in a tree."""
        count = 1  # Count the current node
        for child in node.get_children():
            count += self._count_nodes(child)
        return count
    
    def _format_operations(self, operations, tree1, tree2) -> List[Dict[str, Any]]:
        """Format edit operations into a readable format."""
        formatted_ops = []
        
        for op in operations:
            if op[0] == 'insert':
                node = op[1]
                formatted_ops.append({
                    'operation': 'insert',
                    'path': node.path,
                    'node_type': node.node_type,
                    'value': node.value
                })
            elif op[0] == 'remove':
                node = op[1]
                formatted_ops.append({
                    'operation': 'remove',
                    'path': node.path,
                    'node_type': node.node_type,
                    'value': node.value
                })
            elif op[0] == 'update':
                node1, node2 = op[1], op[2]
                # Get similarity metrics
                sim = self._calculate_node_similarity(node1, node2)
                
                formatted_ops.append({
                    'operation': 'update',
                    'path': node1.path,
                    'from_type': node1.node_type,
                    'to_type': node2.node_type,
                    'from_value': node1.value,
                    'to_value': node2.value,
                    'semantic_similarity': sim["key_sim"] if self.use_semantic_similarity else None,
                    'value_similarity': sim["value_sim"] if sim["is_leaf"] else None
                })
        
        return formatted_ops
    
    def _extract_field_level_metrics(self, operations_by_pair: Dict[Tuple[int, int], List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """
        Extract field-level metrics from edit operations.
        
        Args:
            operations_by_pair: Dictionary mapping pairs of indices to edit operations
            
        Returns:
            Dictionary with field-level metrics
        """
        # Track metrics for each field path
        field_metrics = defaultdict(lambda: {
            "edit_count": 0,
            "update_count": 0,
            "insert_count": 0,
            "remove_count": 0,
            "semantic_similarities": [],
            "value_similarities": [],
            "type_changes": defaultdict(int),
            "presence_count": 0,
            "total_comparisons": 0
        })
        
        # Count total number of comparisons
        total_comparisons = len(operations_by_pair)
        
        # Process all operations
        for ops in operations_by_pair.values():
            # Track which fields were seen in this comparison
            seen_fields = set()
            
            for op in ops:
                op_type = op['operation']
                path = op.get('path', '')
                
                # Skip root node
                if path == 'root':
                    continue
                
                # Normalize path to field level (remove array indices)
                field_path = self._normalize_to_field_path(path)
                
                # Mark this field as seen in this comparison
                seen_fields.add(field_path)
                
                # Update metrics for this field
                metrics = field_metrics[field_path]
                metrics["edit_count"] += 1
                
                if op_type == 'update':
                    metrics["update_count"] += 1
                    
                    # Track semantic similarity if available
                    if op.get('semantic_similarity') is not None:
                        metrics["semantic_similarities"].append(op['semantic_similarity'])
                    
                    # Track value similarity if available
                    if op.get('value_similarity') is not None:
                        metrics["value_similarities"].append(op['value_similarity'])
                    
                    # Track type changes
                    from_type = op.get('from_type', 'unknown')
                    to_type = op.get('to_type', 'unknown')
                    if from_type != to_type:
                        metrics["type_changes"][(from_type, to_type)] += 1
                
                elif op_type == 'insert':
                    metrics["insert_count"] += 1
                
                elif op_type == 'remove':
                    metrics["remove_count"] += 1
            
            # Update presence count for all seen fields
            for field_path in seen_fields:
                field_metrics[field_path]["presence_count"] += 1
        
        # Calculate consistency scores and finalize metrics
        result = {}
        for field_path, metrics in field_metrics.items():
            # Calculate presence consistency (how often the field appears)
            metrics["total_comparisons"] = total_comparisons
            presence_consistency = metrics["presence_count"] / total_comparisons if total_comparisons > 0 else 0
            
            # Calculate value consistency when field is present
            value_similarities = metrics["value_similarities"]
            avg_value_similarity = sum(value_similarities) / len(value_similarities) if value_similarities else 1.0
            
            # Calculate semantic consistency
            semantic_similarities = metrics["semantic_similarities"]
            avg_semantic_similarity = sum(semantic_similarities) / len(semantic_similarities) if semantic_similarities else 1.0
            
            # Calculate overall field consistency score
            # Weight presence more heavily than value consistency
            field_consistency = 0.7 * presence_consistency + 0.3 * avg_value_similarity
            
            # Calculate type consistency
            total_type_changes = sum(metrics["type_changes"].values())
            type_consistency = 1.0 - (total_type_changes / metrics["presence_count"] if metrics["presence_count"] > 0 else 0)
            
            # Add calculated metrics
            result[field_path] = {
                "edit_count": metrics["edit_count"],
                "update_count": metrics["update_count"],
                "insert_count": metrics["insert_count"],
                "remove_count": metrics["remove_count"],
                "presence_count": metrics["presence_count"],
                "presence_consistency": presence_consistency,
                "value_consistency": avg_value_similarity,
                "semantic_consistency": avg_semantic_similarity,
                "type_consistency": type_consistency,
                "field_consistency_score": field_consistency,
                "type_changes": dict(metrics["type_changes"]) if metrics["type_changes"] else {}
            }
        
        return result
    
    def _normalize_to_field_path(self, path: str) -> str:
        """
        Normalize a path to field level by removing array indices.
        
        Args:
            path: The path to normalize
            
        Returns:
            Normalized field path
        """
        # Replace array indices like [0], [1], etc. with [*]
        normalized = re.sub(r'\[\d+\]', '[*]', path)
        return normalized
    
    def evaluate_structural_consistency(self, json_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate structural consistency across multiple JSON outputs with enhanced metrics.
        
        Args:
            json_outputs: List of JSON objects to evaluate
            
        Returns:
            Dictionary with comprehensive consistency metrics
        """
        n = len(json_outputs)
        if n < 2:
            return {
                "error": "Need at least 2 outputs to evaluate consistency",
                "valid_count": n
            }
        
        # Calculate pairwise similarities
        similarities = []
        operations_by_pair = {}
        
        for i in range(n-1):
            for j in range(i+1, n):
                try:
                    sim, ops = self.calculate_tree_edit_distance(json_outputs[i], json_outputs[j])
                    similarities.append((i, j, sim))
                    operations_by_pair[(i, j)] = ops
                except Exception as e:
                    print(f"Error comparing outputs {i} and {j}: {e}")
                    # print error trace
                    import traceback
                    print(f"Error trace:\n{traceback.format_exc()}")
                    operations_by_pair[(i, j)] = []
        
        # Extract similarity values
        similarity_values = [sim for _, _, sim in similarities]
        
        # Calculate basic statistics
        avg_similarity = sum(similarity_values) / len(similarity_values) if similarity_values else 1.0
        std_similarity = float(np.std(similarity_values)) if len(similarity_values) > 1 else 0.0
        min_similarity = min(similarity_values) if similarity_values else 1.0
        max_similarity = max(similarity_values) if similarity_values else 1.0
        similarity_range = max_similarity - min_similarity
        
        # Calculate consistency coefficient (rewards high similarity, penalizes variance)
        normalized_std = min(std_similarity, avg_similarity) / avg_similarity if avg_similarity > 0 else 0
        consistency_coefficient = avg_similarity * (1 - normalized_std)
        
        # Calculate quartile-based metrics
        quartile_metrics = self._calculate_quartile_metrics(similarity_values)
        
        # Calculate entropy-based consistency (if scipy is available)
        entropy_score = self._calculate_similarity_entropy(similarity_values)
        
        # Calculate Gini coefficient for consistency
        gini_coefficient = self._calculate_gini_coefficient(similarity_values)
        
        # Detect outliers using IQR method
        outliers = self._detect_outliers(similarities)
        
        # Find most different pairs
        sorted_similarities = sorted(similarities, key=lambda x: x[2])
        most_different_pairs = sorted_similarities[:3] if len(sorted_similarities) >= 3 else sorted_similarities
        
        # Find most common edit operations
        operation_counts = defaultdict(int)
        path_edit_counts = defaultdict(int)
        
        for ops in operations_by_pair.values():
            for op in ops:
                op_type = op['operation']
                path = op.get('path', '')
                operation_counts[op_type] += 1
                path_edit_counts[path] += 1
        
        # Most frequently edited paths
        frequent_edits = sorted(
            [(path, count) for path, count in path_edit_counts.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]  # Top 10
        
        # Extract field-level metrics
        field_level_metrics = self._extract_field_level_metrics(operations_by_pair)
        
        # Calculate field-level consistency statistics
        field_consistency_scores = [metrics["field_consistency_score"] for metrics in field_level_metrics.values()]
        avg_field_consistency = sum(field_consistency_scores) / len(field_consistency_scores) if field_consistency_scores else 1.0
        
        # Find most and least consistent fields
        sorted_fields = sorted(
            [(field, metrics["field_consistency_score"]) for field, metrics in field_level_metrics.items()],
            key=lambda x: x[1]
        )
        
        least_consistent_fields = sorted_fields[:5] if len(sorted_fields) >= 5 else sorted_fields
        most_consistent_fields = sorted_fields[-5:][::-1] if len(sorted_fields) >= 5 else sorted_fields[::-1]
        
        # Prepare comprehensive report
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "num_outputs_analyzed": n,
            "num_pairwise_comparisons": len(similarities),
            
            # Basic consistency metrics
            "consistency_metrics": {
                "mean_similarity": avg_similarity,
                "std_deviation": std_similarity,
                "min_similarity": min_similarity,
                "max_similarity": max_similarity,
                "similarity_range": similarity_range,
                "consistency_coefficient": consistency_coefficient,
                "perfect_consistency": avg_similarity > 0.99 and std_similarity < 0.01
            },
            
            # Field-level consistency metrics
            "field_level_metrics": {
                "mean_field_consistency": avg_field_consistency,
                "field_count": len(field_level_metrics),
                "most_consistent_fields": [
                    {"field": field, "consistency_score": score}
                    for field, score in most_consistent_fields
                ],
                "least_consistent_fields": [
                    {"field": field, "consistency_score": score}
                    for field, score in least_consistent_fields
                ],
                "detailed_field_metrics": field_level_metrics
            },
            
            # Advanced statistical metrics
            "statistical_metrics": {
                "quartiles": quartile_metrics,
                "entropy": entropy_score,
                "gini_coefficient": gini_coefficient
            },
            
            # Configuration information
            "configuration": {
                "semantic_similarity_enabled": self.use_semantic_similarity,
                "array_order_matters": self.array_order_matters,
                "string_method": self.string_method
            },
            
            # Detailed analysis
            "outliers": [
                {
                    "pair": (i, j),
                    "similarity": sim,
                    "z_score": (sim - avg_similarity) / std_similarity if std_similarity > 0 else 0
                } for i, j, sim in outliers
            ],
            
            "most_different_pairs": [
                {
                    "pair": (i, j),
                    "similarity": sim,
                    "edit_operations": operations_by_pair.get((i, j), [])[:5]  # First 5 operations
                }
                for i, j, sim in most_different_pairs
            ],
            
            "operation_counts": dict(operation_counts),
            
            "frequently_edited_paths": [
                {"path": path, "edit_count": count}
                for path, count in frequent_edits
            ]
        }
        
        return report
    
    def _calculate_quartile_metrics(self, similarities: List[float]) -> Dict[str, float]:
        """Calculate quartile-based metrics from similarity scores."""
        if not similarities:
            return {"q1": 0.0, "median": 0.0, "q3": 0.0, "iqr": 0.0}
        
        sorted_sims = sorted(similarities)
        n = len(sorted_sims)
        
        q1_idx = max(0, n // 4)
        median_idx = max(0, n // 2)
        q3_idx = max(0, (3 * n) // 4)
        
        q1 = sorted_sims[q1_idx]
        median = sorted_sims[median_idx]
        q3 = sorted_sims[q3_idx]
        iqr = q3 - q1
        
        return {
            "q1": q1,
            "median": median,
            "q3": q3,
            "iqr": iqr
        }
    
    def _calculate_similarity_entropy(self, similarities: List[float], bins: int = 10) -> float:
        """Calculate entropy of similarity distribution as a measure of consistency."""
        if not similarities or len(similarities) < 2:
            return 0.0
        
        try:
            # Create histogram of similarities
            hist, _ = np.histogram(similarities, bins=bins, range=(0, 1), density=True)
            
            # Add small epsilon to avoid log(0)
            hist = hist + 1e-10
            hist = hist / hist.sum()
            
            # Calculate entropy
            entropy = -np.sum(hist * np.log2(hist))
            return float(entropy)
        except Exception as e:
            print(f"Error calculating entropy: {e}")
            return 0.0
    
    def _calculate_gini_coefficient(self, similarities: List[float]) -> float:
        """Calculate Gini coefficient for similarity scores."""
        if not similarities:
            return 0.0
        
        # Sort similarities
        sorted_sims = sorted(similarities)
        n = len(sorted_sims)
        
        # Calculate Gini coefficient
        numerator = sum((2 * i - n - 1) * sim for i, sim in enumerate(sorted_sims, 1))
        denominator = n * sum(sorted_sims)
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def _detect_outliers(self, similarities: List[Tuple[int, int, float]], threshold: float = 1.5) -> List[Tuple[int, int, float]]:
        """Detect outliers in similarity scores using IQR method."""
        if len(similarities) < 4:  # Need enough data for quartiles
            return []
        
        # Extract similarity values
        sim_values = [sim for _, _, sim in similarities]
        
        # Calculate quartiles
        sorted_sims = sorted(sim_values)
        n = len(sorted_sims)
        
        q1 = sorted_sims[n // 4]
        q3 = sorted_sims[(3 * n) // 4]
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        # Find outliers
        outliers = [sim_tuple for sim_tuple in similarities 
                   if sim_tuple[2] < lower_bound or sim_tuple[2] > upper_bound]
        
        return outliers

def parse_json_outputs(outputs: List[Union[str, Dict]]) -> List[Dict]:
    """
    Parse a list of JSON outputs that might be strings or dictionaries.
    
    Args:
        outputs: List of JSON strings or dictionaries
        
    Returns:
        List of parsed JSON dictionaries
    """
    parsed = []
    for i, output in enumerate(outputs):
        if isinstance(output, str):
            try:
                parsed.append(json.loads(output))
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse JSON string at index {i}: {e}")
                print(f"String preview: {output[:100]}...")
        elif isinstance(output, list):
            if isinstance(output[0], dict) and len(output)>0:
                parsed.append({"responses": output})
            else:
                print(output[0])
        elif isinstance(output, dict):
            parsed.append(output)
        else:
            print(f"Warning: Skipping non-JSON output at index {i} of type {type(output)}")
    
    return parsed


def evaluate_semantic_json_consistency(
    outputs: List[Union[str, Dict]],
    array_order_matters: bool = True,
    required_fields: List[str] = None,
    use_semantic_similarity: bool = True,
    model_id: str = 'all-MiniLM-L6-v2',
    semantic_threshold: float = 0.7,
    use_hungarian: bool = True,
    long_string_method: str = 'hungarian'
) -> Dict[str, Any]:
    """
    Evaluate structural consistency of JSON outputs with semantic similarity support.
    
    Args:
        outputs: List of JSON strings or dictionaries
        array_order_matters: Whether array order matters for comparison
        required_fields: List of required field paths
        use_semantic_similarity: Whether to use embedding-based semantic similarity
        model_id: Name of the sentence transformer model or Bedrock model ID
        semantic_threshold: Minimum semantic similarity to consider keys as matching
        use_hungarian: Whether to use Hungarian algorithm for optimal matching
        long_string_method: Method for long string comparison
        
    Returns:
        Dictionary with consistency metrics
    """
    # Parse outputs
    parsed_outputs = parse_json_outputs(outputs)
    
    if len(parsed_outputs) < 2:
        return {
            "error": "Need at least 2 valid JSON outputs to evaluate consistency",
            "valid_count": len(parsed_outputs),
            "total_count": len(outputs)
        }
    
    # Create evaluator
    evaluator = SemanticJsonTreeConsistencyEvaluator(
        array_order_matters=array_order_matters,
        required_fields=set(required_fields) if required_fields else set(),
        use_semantic_similarity=use_semantic_similarity,
        model_id=model_id,
        semantic_threshold=semantic_threshold,
        use_hungarian=use_hungarian,
        long_string_method=long_string_method
    )
    
    # Evaluate consistency
    return evaluator.evaluate_structural_consistency(parsed_outputs)


if __name__ == "__main__":
    # Example usage
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
    
    json3 = {
        "firstName": "John",
        "lastName": "Doe",
        "contact": {
            "email": "john.doe@example.com",
            "phone": "123-456-7890"
        }
    }
    
    print("=== Semantic JSON Tree Consistency Evaluation ===\n")
    
    # Test with semantic similarity enabled
    print("1. With Semantic Similarity:")
    result_semantic = evaluate_semantic_json_consistency(
        [json1, json2, json3],
        array_order_matters=False,
        use_semantic_similarity=True,
        semantic_threshold=0.6,
        use_hungarian=True,
        long_string_method='hungarian',
        model_id="all-MiniLM-L6-v2"  # Changed from Bedrock model for example
    )
    
    print(f"   Consistency Metrics: {result_semantic['consistency_metrics']}")
    
    # Test without semantic similarity
    print("\n2. Without Semantic Similarity:")
    result_exact = evaluate_semantic_json_consistency(
        [json1, json2, json3],
        array_order_matters=False,
        use_semantic_similarity=False,
        use_hungarian=False,
        long_string_method='direct'
    )
    
    print(f"   Consistency Metrics: {result_exact['consistency_metrics']}")
    
    # Compare the results
    print("\n3. Improvement with Semantic Similarity:")
    semantic_score = result_semantic['consistency_metrics']['mean_similarity']
    exact_score = result_exact['consistency_metrics']['mean_similarity']
    improvement = semantic_score - exact_score
    print(f"   Consistency Score Improvement: {improvement:.4f} ({improvement*100:.1f}%)")
    
    # Show most different pairs
    print("\n4. Most Different Pairs (with Semantic Similarity):")
    for pair_info in result_semantic["most_different_pairs"]:
        print(f"   Pair {pair_info['pair']}: Similarity {pair_info['similarity']:.4f}")
        
    # Show frequently edited paths
    print("\n5. Frequently Edited Paths (with Semantic Similarity):")
    for path_info in result_semantic["frequently_edited_paths"][:5]:
        print(f"   {path_info['path']}: {path_info['edit_count']} edits")