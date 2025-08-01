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
from functools import lru_cache
import warnings
import re
import time
from bert_score import score as bert_score

from scipy.optimize import linear_sum_assignment
import zss

# Optional imports with proper error handling
from langchain.text_splitter import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer
import torch

import boto3
from botocore.exceptions import ClientError
from botocore.config import Config

from deepdiff import DeepDiff

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import concurrent.futures

import itertools
import math

from transformers import logging
logging.set_verbosity_error()

def getEmbeddings(text, model_id, bedrock_client, max_retries=10, initial_delay=2, output_embedding_length=1024):
    """
    Get embeddings from Bedrock with proper retry and connection handling
    """
    # Validate input text
    if not text or not text.strip():
        warnings.warn(f"Empty or whitespace-only text provided for embedding: '{text}'")
        return np.zeros(output_embedding_length)
    
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

class StringSimilarityCache:
    def __init__(self):
        self.cache = {}
    
    def get_key(self, s1: str, s2: str) -> str:
        return "|".join(sorted([s1, s2]))
    
    def get(self, s1: str, s2: str) -> Optional[float]:
        return self.cache.get(self.get_key(s1, s2))
    
    def set(self, s1: str, s2: str, score: float):
        self.cache[self.get_key(s1, s2)] = score
    
    def batch_set(self, pairs: List[Tuple[str, str]], scores: List[float]):
        for (s1, s2), score in zip(pairs, scores):
            self.set(s1, s2, score)

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
                 original_zss: bool = False,
                 path_weight_decay: float = 0.9,
                 type_change_cost: Dict[Tuple[str, str], float] = None,
                 required_fields: Set[str] = None,
                 model_id: str = 'all-MiniLM-L6-v2',
                 chunk_size: int = 300,
                 chunk_overlap: int = 50):
        """
        Initialize the evaluator with semantic capabilities.
        
        Args:
            original_zss: whether using original zss
            path_weight_decay: Weight decay factor for deeper paths (0-1)
            type_change_cost: Custom costs for type changes
            required_fields: Set of required field paths
            model_id: Name of the sentence transformer model or Bedrock model ID
            chunk_size: Size of chunks for text splitting
            chunk_overlap: Overlap between chunks
        """
        self.path_weight_decay = path_weight_decay
        self.type_change_cost = type_change_cost or self._default_type_change_costs()
        self.required_fields = required_fields or set()
        
        self.cache = StringSimilarityCache()
        
        # Initialize embedding model if available
        self.embedding_model = None
        self.bedrock_client = None
        self.model_id = model_id
        
        if self.model_id in ["amazon.titan-embed-text-v1", "amazon.titan-embed-text-v2:0", "cohere.embed-multilingual-v3"]:
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
                config=bedrock_config,
                region_name="us-west-2"
            )
        else:
            self.embedding_model = SentenceTransformer(self.model_id)
            # Warm up the model
            self.embedding_model.encode(["test"], show_progress_bar=False)
        
        # Cache for embeddings
        self._embedding_cache = {}
        
        # Text splitting configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.calculate_similarity_method = {
            "ted": self.calculate_tree_edit_distance,
            "bertscore": self.calculate_bertscore,
            "deepdiff": self.calculate_similarity_with_deepdiff
        }
        
        self.weights = {
            'type': 0.1,
            'value': 0.8,
            'key': 0.1
        }
        
        self.original_zss = original_zss
        self.batch_size_bertscore = 2000
        
    def set_original_zss(self, original_zss: bool=False):
        self.original_zss = original_zss
    
    def collect_all_string_pairs(self, json_outputs: Union[Dict, List[Dict]], gt: Union[Dict, List[Dict], None] = None) -> List[Tuple[str, str]]:
        # get all values from json_outputs
        output_values = self.collect_all_values(json_outputs)
        
        pairs = []
        seen = set()
            
        ref_values = self.collect_all_values(gt) if gt else output_values.copy()
            
        # create all pairs from output_values and gt_values
        for item1 in ref_values:
            for item2 in output_values:
                # Sort the pair to ensure consistent ordering
                pair_tuple = tuple(sorted([str(item1), str(item2)]))
                if pair_tuple not in seen:
                    seen.add(pair_tuple)
                    pairs.append((item1, item2))
        return pairs
    
    def collect_all_values(self, data: Union[Dict, List, Any]) -> List[Tuple[str, Any]]:
        """
        Collect all values from a nested dictionary/list structure.
        
        Args:
            data: The input data structure (dict, list, or primitive)
            include_keys: If True, also collect dictionary keys as values
            
        Returns:
            List of tuples (path, value) where path shows the location of the value
        """
        values = []
        def _collect_recursive(obj):
            """Recursively collect values with their paths."""
            if isinstance(obj, dict):
                # Collect dictionary values
                for key, value in obj.items():
                    _collect_recursive(value)
                    
            elif isinstance(obj, list):
                # Collect list elements
                for idx, item in enumerate(obj):
                    _collect_recursive(item)
                    
            else:
                # Leaf value (string, number, boolean, None, etc.)
                if isinstance(obj, str):
                    values.append(obj)
        
        _collect_recursive(data)
        return values
    
    def batch_compute_similarities(self, pairs: List[Tuple[str, str]], method='cosine') -> Dict[Tuple[str, str], float]:
        """Batch compute BERT scores for all unique pairs."""
        uncached_pairs = [(s1, s2) for s1, s2 in pairs if self.cache.get(s1, s2) is None]
    
        if not uncached_pairs:
            # Return cached values for requested pairs
            return {(s1, s2): self.cache.get(s1, s2) for s1, s2 in pairs 
                    if self.cache.get(s1, s2) is not None}
        
        batch_size = min(self.batch_size_bertscore, len(uncached_pairs))
        # Process pairs by batch
        for i in range(0, len(uncached_pairs), batch_size):
            batch = uncached_pairs[i:i+batch_size]
            refs, cands = zip(*batch)
            #P, R, F1 = bert_score(list(cands), list(refs), lang="en", verbose=False)
            #scores = [float(f.item()) for f in F1]
            if method == "bertscore":
                P, R, F1 = bert_score(list(cands), list(refs), lang="en", verbose=False)
                scores = [float(f.item()) for f in F1]
            else:
                scores = [self._calculate_semantic_similarity(s1, s2) for s1, s2 in zip(list(cands), list(refs))]
                
            
            self.cache.batch_set(batch, scores)
        
        return self.cache.cache
        
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
        costs[("object", "array")] = costs[("array", "object")] = 1
        costs[("object", "string")] = costs[("string", "object")] = 1
        costs[("object", "number")] = costs[("number", "object")] = 1
        costs[("array", "string")] = costs[("string", "array")] = 1
        costs[("array", "number")] = costs[("number", "array")] = 1
        
        return costs
    
    @lru_cache(maxsize=2000)
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for a text with caching."""
        
        if self.bedrock_client:
            try:
                return getEmbeddings(text, self.model_id, self.bedrock_client)
                #return [1,2,3,4,5]
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
        if text1 == text2:
            return 1.0
        
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
    
    def _calculate_node_similarity(self, node1: JsonNode, node2: JsonNode, key_sim_threshold: float = 0.8) -> Dict[str, Any]:
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
        
        # Calculate key similarity
        key_sim = self._calculate_semantic_similarity(key1, key2)
        
        value_cost = 1.0
                
        #string_pairs = []
        if key_sim > key_sim_threshold:
            # Calculate value similarity for leaf nodes
            if not node1.children and not node2.children and type_match:
                if node1.node_type == "string":
                    #string_pairs.append((str(node1.value), str(node2.value)))
                    value_cost = 1 - self._compare_strings(str(node1.value), str(node2.value))
                elif node1.node_type == "number":
                    value_cost = 1 - self._compare_numbers(float(node1.value), float(node2.value))
                elif node1.node_type == "array":
                    value_cost = self._compare_arrays_unordered(list(node1.value), list(node2.value), node1.label, node2.label)
                elif node1.value == node2.value:  # For boolean, null, etc.
                    value_cost = 0.0
        
        #print(f"node1.node_type: {node1.node_type}, value_sim: {value_sim}") 
        return {
            "type_match": type_match,
            "path_match": path_match,
            "key_sim": key_sim,
            "value_cost": value_cost,
            "key1": key1,
            "key2": key2,
            "is_leaf": not node1.children and not node2.children
        }
    
    def _compare_strings(self, str1: str, str2: str, method="cosine") -> float:
        """Compare two strings with optional semantic similarity and chunking for long text."""
        # Quick equality check
        
        if str1 == str2:
            return 1.0
        
        # Check cache first
        cached = self.cache.get(str1, str2)
        if cached is not None:
            #print(f"str1: {str1}, str2: {str2}, sim: {cached}")
            return cached
        
        if len(str1) < self.chunk_size and len(str2) < self.chunk_size:
            if method == "bertscore":
                P, R, F1 = bert_score([str1], [str2], lang="en")
                sim = float(F1.item())
            else:
                sim = self._calculate_semantic_similarity(str1, str2)
                # Apply domain penalty for unrelated content
                sim = self._apply_domain_penalty(str1, str2, sim)
            return sim
        else:
            return self._compare_long_strings_hungarian(str1, str2)
    
    def _apply_domain_penalty(self, str1: str, str2: str, similarity: float) -> float:
        """
        Apply domain-specific penalty for unrelated content.
        
        Args:
            str1: First string
            str2: Second string  
            similarity: Base similarity score
            
        Returns:
            Adjusted similarity score
        """
        # Simple implementation - can be enhanced with domain-specific logic
        if not str1.strip() or not str2.strip():
            return 0.0
        
        # Apply penalty for very different string lengths
        len_ratio = min(len(str1), len(str2)) / max(len(str1), len(str2))
        if len_ratio < 0.3:  # Very different lengths
            similarity *= 0.8
            
        return similarity
    
    def _compare_long_strings_hungarian(self, str1: str, str2: str) -> float:
        """Compare long strings by breaking them into chunks and using Hungarian algorithm for optimal matching."""
        # Split into chunks (sentences or paragraphs)
        chunks1 = self._split_into_chunks(str1)
        chunks2 = self._split_into_chunks(str2)
        
        # Create similarity matrix between all chunks
        similarity_matrix = np.zeros((len(chunks1), len(chunks2)))
        
        
        #string_pairs = []
        for i, chunk1 in enumerate(chunks1):
            for j, chunk2 in enumerate(chunks2):
                # Use appropriate method for chunk comparison
                similarity_matrix[i, j] = self._compare_strings(chunk1, chunk2)
                #string_pairs.append((chunk1, chunk2))
        
        #print(f"_compare_long_strings_hungarian: {len(string_pairs)}")
        
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
        
        # Return the better of the two approaches
        return chunk_sim  # Slight penalty for direct comparison
    
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
        if len(text) < self.chunk_size:
            return [text]
        
        # Create a text splitter that tries to create semantically meaningful chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]  # Try these separators in order
        )
        
        # Split the text
        return text_splitter.split_text(text)
    
    def _compare_numbers(self, num1: float, num2: float) -> float:
        """Compare two numbers with tolerance."""
        return 0 if num1 != num2 else 1
    
    def _count_json_elements(self, json_obj: Any) -> int:
        """
        Count the total number of elements in a JSON structure.
        Used for normalizing structural penalties.
        
        Args:
            json_obj: JSON object to count elements in
            
        Returns:
            Total number of elements (keys, values, array items)
        """
        if isinstance(json_obj, dict):
            return len(json_obj) + sum(self._count_json_elements(v) for v in json_obj.values())
        elif isinstance(json_obj, list):
            return len(json_obj) + sum(self._count_json_elements(item) for item in json_obj)
        else:
            return 1  # Primitive value counts as 1 element
            
    def update_cost(self, node1: JsonNode, node2: JsonNode) -> float:
        """
        Calculate the cost of updating a node.
        
        Args:
            node1: The source node
            node2: The target node
            
        Returns:
            The cost of update
        """
        # Quick check for identical nodes
        if (node1.label == node2.label and 
            node1.node_type == node2.node_type and 
            node1.value == node2.value and
            len(node1.children) == len(node2.children)):
            #print(f"update_cost: {node1.label}->{node1.value}, {node2.label}->{node2.value} cost: 0")
            return 0.0  # Identical nodes have zero update cost
        
        # Get similarity metrics from shared calculation
        sim = self._calculate_node_similarity(node1, node2)
        #print(f"update_cost-> sim: {sim}")
        # Base cost for type change
        type_cost = self.type_change_cost.get(
            (node1.node_type, node2.node_type), 
            1.0  # Default cost if not specified
        )
        
        # Calculate value cost
        if sim["type_match"] and sim["is_leaf"]:
            # For leaf nodes with same type, use the value similarity
            value_cost = sim['value_cost']
        else:
            value_cost = 1.0  # No value cost if types are different or not leaf nodes
        
        
        # Calculate key difference factor
        key_cost = 1 - sim["key_sim"]
        
        # 特殊情况处理
        if sim["type_match"] and sim["is_leaf"] and sim["value_cost"] == 0 and sim["key_sim"] == 1.0:
            #print(f"update_cost: {node1.label}->{node1.value}, {node2.label}->{node2.value} cost: 0")
            return 0.0  # 完全相同
        
        #print(f"type_cost: {type_cost}, value_cost: {value_cost}, key_cost: {key_cost}")
        
        # 加权平均
        cost = (
            self.weights['type'] * type_cost +
            self.weights['value'] * value_cost +
            self.weights['key'] * key_cost
        )
        
        # Apply path-based weighting - consider both paths
        # Use the average of both path weights for symmetry
        path_weight1 = self.calculate_path_weight(node1.path)
        path_weight2 = self.calculate_path_weight(node2.path)
        avg_path_weight = (path_weight1 + path_weight2) / 2.0
        
        cost *= avg_path_weight
        return cost
    
    def _compare_arrays_unordered(self, arr1: List[Any], arr2: List[Any], arr1_label: str, arr2_label: str) -> float:
        """Compare arrays without considering order using optimal matching."""
        if len(arr1) == 0 and len(arr2) == 0:
            return 0  # Both empty arrays are identical
        if len(arr1) == 0 or len(arr2) == 0:
            return 1  # One empty, one not
        
        # Create similarity matrix
        cost_matrix = np.ones((len(arr1), len(arr2)))
        
        for i, item1 in enumerate(arr1):
            for j, item2 in enumerate(arr2):
                # Handle different types appropriately
                if type(item1) != type(item2):
                    cost_matrix[i, j] = 1.0
                elif isinstance(item1, str):
                    cost_matrix[i, j] = 1-self._compare_strings(str(item1), str(item2))
                elif isinstance(item1, (int, float)):
                    cost_matrix[i, j] = 1-self._compare_numbers(float(item1), float(item2))
                elif isinstance(item1, dict):
                    # Recursive comparison for nested objects
                    tree1 = self.json_to_tree(item1, f"{arr1_label}")
                    tree2 = self.json_to_tree(item2, f"{arr2_label}")
                    cost_matrix[i, j] = self._calculate_optimal_matching_cost(tree1, tree2)
                elif isinstance(item1, list):
                    # Recursive array comparison
                    cost_matrix[i, j] = self._compare_arrays_unordered(item1, item2, arr1_label, arr2_label)
                elif item1 and item1 == item2:
                    cost_matrix[i, j] = 0.0
                else:
                    cost_matrix[i, j] = 1.0
        
        # Use Hungarian algorithm for optimal matching
        if cost_matrix.shape[0] != cost_matrix.shape[1]:
            # Pad matrix for Hungarian algorithm
            max_len = max(len(arr1), len(arr2))
            padded_matrix = np.ones((max_len, max_len))
            padded_matrix[:len(arr1), :len(arr2)] = cost_matrix
            sim_matrix = padded_matrix
        
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Calculate total cost including unmatched elements
        total_cost = 0
        matched_elements = 0
        
        len1, len2 = len(arr1), len(arr2)
        
        # Create a square cost matrix for Hungarian algorithm
        max_len = max(len1, len2)
        square_matrix = np.ones((max_len, max_len))  # Initialize with 1s (full cost for unmatched)
        square_matrix[:len1, :len2] = cost_matrix
        
        for i, j in zip(row_indices, col_indices):
            if i < len1 and j < len2:
                # This is a matched pair from the original arrays
                total_cost += square_matrix[i, j]
                matched_elements += 1
            elif i < len1:
                # Element from arr1 that couldn't be matched (deleted)
                total_cost += 1.0  # Full deletion cost
            elif j < len2:
                # Element from arr2 that couldn't be matched (inserted)
                total_cost += 1.0  # Full insertion cost
        
        # Also account for any unmatched elements not covered by the assignment
        # (This handles cases where one array is larger)
        unmatched_from_arr1 = len1 - matched_elements
        unmatched_from_arr2 = len2 - matched_elements
        total_cost += max(0, unmatched_from_arr1) + max(0, unmatched_from_arr2)
        
        # Normalize by the maximum array length
        normalized_cost = total_cost / max_len if max_len > 0 else 0
        
        return normalized_cost
    
    def _calculate_optimal_matching_cost(
        self, tree1: JsonNode, tree2: JsonNode
    ) -> float:
        """
        Calculate optimal matching cost between two trees using Hungarian algorithm.
        This addresses the issue where tree edit distance makes suboptimal choices due to ordering.
        """
        # Base case: both are leaf nodes
        if not tree1.children and not tree2.children:
            return self.update_cost(tree1, tree2)

        # If one is leaf and other is not, use insert/delete costs
        if not tree1.children and tree2.children:
            return self.delete_cost(tree1) + sum(
                self.insert_cost(child) for child in tree2.children
            )
        if tree1.children and not tree2.children:
            return sum(
                self.delete_cost(child) for child in tree1.children
            ) + self.insert_cost(tree2)

        # Both have children - use Hungarian algorithm for optimal matching
        children1 = tree1.children
        children2 = tree2.children
        
        if not children1 and not children2:
            return self.update_cost(tree1, tree2)

        # Create cost matrix for Hungarian algorithm
        n1, n2 = len(children1), len(children2)
        max_size = max(n1, n2)

        # Pad to square matrix
        cost_matrix = np.full((max_size, max_size), float("inf"))

        # Fill actual costs
        for i in range(n1):
            for j in range(n2):
                # Cost of matching child i with child j
                cost_matrix[i][j] = self._calculate_optimal_matching_cost(
                    children1[i], children2[j]
                )

        # Cost of unmatched nodes (insert/delete)
        for i in range(n1, max_size):
            for j in range(n2):
                cost_matrix[i][j] = self.insert_cost(children2[j])

        for i in range(n1):
            for j in range(n2, max_size):
                cost_matrix[i][j] = self.delete_cost(children1[i])

        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Calculate total cost
        total_cost = self.update_cost(tree1, tree2)  # Cost of updating root nodes
        for i, j in zip(row_indices, col_indices):
            total_cost += cost_matrix[i][j]
        
        #print(f"total_cost: {total_cost}")
        return total_cost
    
    def calculate_tree_edit_distance(self, json1: Dict[str, Any], json2: Dict[str, Any]) -> float:
        """
        Calculate tree edit distance between two JSON objects.
        
        Args:
            json1: First JSON object
            json2: Second JSON object
            
        Returns:
            similarity_score
        """
        # Convert JSONs to trees
        tree1 = self.json_to_tree(json1)
        tree2 = self.json_to_tree(json2)
        
        # Use Zhang-Shasha algorithm from zss
        # Check which version of zss API we're using
        
        if self.original_zss:
            try:
                # Newer zss versions
                distance = zss.distance(
                    tree1, tree2,
                    get_children=lambda x: x.get_children(),
                    insert_cost=self.insert_cost,
                    remove_cost=self.delete_cost,
                    update_cost=self.update_cost
                )
            except TypeError as e:
                raise TypeError(
                    f"Failed to calculate tree distance. Ensure zss is properly installed "
                    f"and trees have compatible structure: {str(e)}"
                ) from e
        else:
            distance = self._calculate_optimal_matching_cost(tree1, tree2)
            
        # Calculate tree sizes for normalization
        size1 = self._count_nodes(tree1)
        size2 = self._count_nodes(tree2)
        
        max_cost = size1 + size2
        
        # Convert distance to similarity
        if max_cost == 0:
            return 1.0  # Identical empty structures
        
        # Normalize distance and convert to similarity
        normalized_distance = distance / max_cost
        similarity = 1.0 - normalized_distance
        
        # Ensure similarity is in [0, 1] range
        return max(0.0, min(1.0, similarity))
        
    def _count_nodes(self, node: JsonNode) -> int:
        """Count the number of nodes in a tree."""
        count = 1 if node.value else 0 # Count the current node
        for child in node.get_children():
            count += self._count_nodes(child)
        
        if node.node_type == "array" and len(node.value) > 0 and isinstance(node.value[0],dict):
            for el in node.value:
                el_tree = self.json_to_tree(el)
                count += self._count_nodes(el_tree)
        return count
    
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
    
    def calculate_bertscore(self, json1: Dict[str, Any], json2: Dict[str, Any], **kwargs) -> float:
        P, R, F1 = bert_score([str(json1)], [str(json2)], lang="en")
        return float(F1.item())
    
    def calculate_similarity_with_deepdiff(self, json1: [Dict[str, Any], List], json2: [Dict[str, Any], List], **kwargs) -> float:
        """
        Calculate similarity using DeepDiff with enhanced value comparison.
        Uses semantic similarity for strings and proper comparison for numbers.
        
        Args:
            json1: First JSON object
            json2: Second JSON object
            
        Returns:
            Similarity score between 0 and 1 (1 = identical, 0 = completely different)
        """
        try:
            # Use DeepDiff to find structural differences
            diff = DeepDiff(json1, json2, ignore_order=True, cache_size=5000)
            
            # If no differences found, return perfect similarity
            if not diff:
                return 1.0
            
            # Calculate similarity based on different types of changes
            total_similarity_score = 0.0
            total_comparisons = 0
            
            # Handle value changes with semantic comparison
            if 'values_changed' in diff:
                for path, change in diff['values_changed'].items():
                    old_value = change['old_value']
                    new_value = change['new_value']
                    
                    # Use appropriate comparison method based on value types
                    if isinstance(old_value, str) and isinstance(new_value, str):
                        # Use semantic string comparison
                        value_similarity = self._compare_strings(old_value, new_value)
                    elif isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                        # Use number comparison
                        value_similarity = self._compare_numbers(float(old_value), float(new_value))
                    elif isinstance(old_value, bool) and isinstance(new_value, bool):
                        # Boolean comparison
                        value_similarity = 1.0 if old_value == new_value else 0.0
                    elif old_value is None or new_value is None:
                        # Null comparison
                        value_similarity = 1.0 if old_value == new_value else 0.0
                    else:
                        # Fallback for other types (convert to string and compare)
                        value_similarity = self._compare_strings(str(old_value), str(new_value))
                    
                    total_similarity_score += value_similarity
                    total_comparisons += 1
            
            # Handle structural changes (additions/removals) - these get lower similarity
            structural_changes = 0
            
            if 'dictionary_item_added' in diff:
                structural_changes += len(diff['dictionary_item_added'])
            
            if 'dictionary_item_removed' in diff:
                structural_changes += len(diff['dictionary_item_removed'])
            
            if 'iterable_item_added' in diff:
                structural_changes += len(diff['iterable_item_added'])
            
            if 'iterable_item_removed' in diff:
                structural_changes += len(diff['iterable_item_removed'])
            
            if 'type_changes' in diff:
                # Type changes are significant structural differences
                structural_changes += len(diff['type_changes']) * 2
            
            # Calculate overall similarity using a change-ratio approach
            if total_comparisons == 0 and structural_changes == 0:
                return 1.0  # No changes found
            
            # Count total comparable elements in both JSONs
            total_elements_json1 = self._count_json_elements(json1)
            total_elements_json2 = self._count_json_elements(json2)
            
            # Use the larger JSON size as the base for comparison
            # This ensures we account for all elements in the comparison
            base_elements = max(total_elements_json1, total_elements_json2)
            
            if base_elements == 0:
                return 1.0
            
            # Calculate similarity based on the proportion of changes
            total_changes = total_comparisons + structural_changes
            
            # If we have value changes, use their average similarity
            # If we have only structural changes, they contribute 0 similarity
            if total_comparisons > 0:
                avg_change_similarity_value = total_similarity_score / total_comparisons
            else:
                avg_change_similarity_value = 0.0  # Only structural changes
            
            
            # Calculate change ratios
            change_ratio_value = min(total_comparisons / base_elements, 1.0)
            change_ratio_schema = min(structural_changes / base_elements, 1.0)
            
            avg_change_similarity_value = (1-change_ratio_value) * avg_change_similarity_value
            schema_similarity_value = 1 - change_ratio_schema
            
            # Overall similarity calculation:
            overall_similarity = avg_change_similarity_value * 0.3 + schema_similarity_value * 0.7
            
            # Ensure similarity is in valid range
            return max(0.0, min(1.0, overall_similarity))
            
        except Exception as e:
            # If DeepDiff fails, fall back to simple comparison
            warnings.warn(f"DeepDiff calculation failed: {e}")
            return 1.0 if json1 == json2 else 0.0
    

    
    def evaluate_structural_consistency(self, json_outputs: List[Dict[str, Any]], gt: Dict[str, Any]=None, method_name: str="ted") -> Dict[str, Any]:
        """
        Evaluate structural consistency across multiple JSON outputs with enhanced metrics.
        
        Args:
            json_outputs: List of JSON objects to evaluate
            
        Returns:
            Dictionary with comprehensive consistency metrics
        """
        
        n = len(json_outputs)
        if gt is None and n < 2:
            return {
                "error": "Need at least 2 outputs to evaluate consistency",
                "valid_count": n
            }
        
        if gt:
            all_pairs = self.collect_all_string_pairs(json_outputs, gt)
            self.batch_compute_similarities(all_pairs)
            
            similarity_values = [self.calculate_similarity_method[method_name](gt, json_output) for json_output in json_outputs]
        else:
            all_pairs = self.collect_all_string_pairs(json_outputs)
            
            self.batch_compute_similarities(all_pairs)
            
            similarity_values = []
            for i in range(n-1):
                for j in range(i+1, n):
                    sim = self.calculate_similarity_method[method_name](json_outputs[i], json_outputs[j])
                    similarity_values.append(sim)

        # Calculate basic statistics
        avg_similarity = sum(similarity_values) / len(similarity_values) if similarity_values else 1.0
        std_similarity = float(np.std(similarity_values)) if len(similarity_values) > 1 else 0.0
        min_similarity = min(similarity_values) if similarity_values else 1.0
        max_similarity = max(similarity_values) if similarity_values else 1.0
        similarity_range = max_similarity - min_similarity
        
        # Calculate consistency coefficient (rewards high similarity, penalizes variance)
        normalized_std = min(std_similarity, avg_similarity) / avg_similarity if avg_similarity > 0 else 0
        consistency_coefficient = avg_similarity * (1 - normalized_std)
        
        # Prepare comprehensive report
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "num_outputs_analyzed": n,
            
            # Basic consistency metrics
            "consistency_metrics": {
                "mean_similarity": avg_similarity,
                "std_deviation": std_similarity,
                "min_similarity": min_similarity,
                "max_similarity": max_similarity,
                "similarity_range": similarity_range,
            }
        }
        
        return report

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
            elif isinstance(output[0], str) and len(output)>0:
                parsed.append({"responses": json.loads(str)})
            else:
                parsed.append({"responses": output})
        elif isinstance(output, dict):
            parsed.append(output)
        else:
            parsed.append({"responses": output})
    
    return parsed

if __name__ == "__main__":
    # Example usage
    evaluator = SemanticJsonTreeConsistencyEvaluator(
        model_id='amazon.titan-embed-text-v2:0',
    )
    
    #data = [[{'question': 'What threshold was used to reclassify flood fraction composites to binary in the analysis?', 'options': [{'answer': '20 percent', 'is_correct': True}, {'answer': '5 percent', 'is_correct': False}, {'answer': '10 percent', 'is_correct': False}, {'answer': '15 percent', 'is_correct': False}]}, {'question': 'Which percentile range was used for the OND 2024 flood exposure estimates?', 'options': [{'answer': '25-75th percentile', 'is_correct': True}, {'answer': '50-95th percentile', 'is_correct': False}, {'answer': '10-50th percentile', 'is_correct': False}, {'answer': '75-95th percentile', 'is_correct': False}]}, {'question': 'What population dataset was used in the analysis?', 'options': [{'answer': 'WorldPop (2020 UN Adjusted)', 'is_correct': True}, {'answer': 'WorldPop (2022 UN Adjusted)', 'is_correct': False}, {'answer': 'WorldPop (2019 UN Adjusted)', 'is_correct': False}, {'answer': 'WorldPop (2021 UN Adjusted)', 'is_correct': False}]}, {'question': 'What time period of FloodScan data was analyzed?', 'options': [{'answer': '1998-2022', 'is_correct': True}, {'answer': '2000-2022', 'is_correct': False}, {'answer': '1995-2022', 'is_correct': False}, {'answer': '2010-2022', 'is_correct': False}]}, {'question': 'Which seasonal forecast influenced the choice of percentile range for MAM 2024?', 'options': [{'answer': 'ECMWF prediction of above average precipitation', 'is_correct': True}, {'answer': 'ECMWF prediction of below average precipitation', 'is_correct': False}, {'answer': 'NOAA prediction of above average precipitation', 'is_correct': False}, {'answer': 'WMO prediction of above average precipitation', 'is_correct': False}]}], [{'question': 'What population dataset was used in combination with FloodScan data for the flood exposure analysis?', 'options': [{'answer': 'WorldPop (2020 UN Adjusted)', 'is_correct': True}, {'answer': 'UNFPA 2024 Dataset', 'is_correct': False}, {'answer': 'UN OCHA 2023 Population Data', 'is_correct': False}, {'answer': 'Somalia Census 2022', 'is_correct': False}]}, {'question': 'What threshold was used to reclassify the flood fraction composites to binary?', 'options': [{'answer': '20 percent', 'is_correct': True}, {'answer': '10 percent', 'is_correct': False}, {'answer': '15 percent', 'is_correct': False}, {'answer': '25 percent', 'is_correct': False}]}, {'question': 'Which percentile range was used for the OND 2024 flood exposure predictions?', 'options': [{'answer': '25-75th percentile', 'is_correct': True}, {'answer': '50-95th percentile', 'is_correct': False}, {'answer': '10-90th percentile', 'is_correct': False}, {'answer': '30-70th percentile', 'is_correct': False}]}, {'question': 'What years of historical FloodScan data were analyzed?', 'options': [{'answer': '1998-2022', 'is_correct': True}, {'answer': '2000-2022', 'is_correct': False}, {'answer': '1995-2022', 'is_correct': False}, {'answer': '2010-2022', 'is_correct': False}]}, {'question': 'What was the minimum flood fraction value threshold used in the masked analysis?', 'options': [{'answer': '0.05 percent', 'is_correct': True}, {'answer': '0.1 percent', 'is_correct': False}, {'answer': '0.5 percent', 'is_correct': False}, {'answer': '1 percent', 'is_correct': False}]}], [{'question': 'What threshold was used to reclassify flood fraction composites to binary in the analysis?', 'options': [{'answer': '20 percent', 'is_correct': True}, {'answer': '5 percent', 'is_correct': False}, {'answer': '10 percent', 'is_correct': False}, {'answer': '15 percent', 'is_correct': False}]}, {'question': 'Which percentile range was used for the OND 2024 flood exposure estimates?', 'options': [{'answer': '25-75th percentile', 'is_correct': True}, {'answer': '50-95th percentile', 'is_correct': False}, {'answer': '10-50th percentile', 'is_correct': False}, {'answer': '75-95th percentile', 'is_correct': False}]}, {'question': 'What population dataset was used in the flood exposure analysis?', 'options': [{'answer': 'WorldPop (2020 UN Adjusted)', 'is_correct': True}, {'answer': 'WorldPop (2022 UN Adjusted)', 'is_correct': False}, {'answer': 'WorldPop (2019 UN Adjusted)', 'is_correct': False}, {'answer': 'WorldPop (2021 UN Adjusted)', 'is_correct': False}]}, {'question': 'What time period of FloodScan data was analyzed in the study?', 'options': [{'answer': '1998-2022', 'is_correct': True}, {'answer': '2000-2022', 'is_correct': False}, {'answer': '1995-2022', 'is_correct': False}, {'answer': '2010-2022', 'is_correct': False}]}, {'question': 'Which seasonal forecast influenced the choice of percentile range for MAM 2024?', 'options': [{'answer': 'ECMWF prediction of above average precipitation', 'is_correct': True}, {'answer': 'ECMWF prediction of below average precipitation', 'is_correct': False}, {'answer': 'NOAA prediction of above average precipitation', 'is_correct': False}, {'answer': 'WMO prediction of average precipitation', 'is_correct': False}]}], [{'question': 'What threshold was used to reclassify flood fraction composites to binary in the analysis?', 'options': [{'answer': '20 percent', 'is_correct': True}, {'answer': '5 percent', 'is_correct': False}, {'answer': '10 percent', 'is_correct': False}, {'answer': '15 percent', 'is_correct': False}]}, {'question': 'Which percentile range was used for the OND 2024 flood exposure estimates?', 'options': [{'answer': '25-75th percentile', 'is_correct': True}, {'answer': '50-95th percentile', 'is_correct': False}, {'answer': '10-50th percentile', 'is_correct': False}, {'answer': '75-95th percentile', 'is_correct': False}]}, {'question': 'What population dataset was used in the flood exposure analysis?', 'options': [{'answer': 'WorldPop (2020 UN Adjusted)', 'is_correct': True}, {'answer': 'WorldPop (2022 UN Adjusted)', 'is_correct': False}, {'answer': 'UN OCHA Population Data 2023', 'is_correct': False}, {'answer': 'UNFPA Census Data 2021', 'is_correct': False}]}, {'question': 'What time period of FloodScan data was analyzed in the study?', 'options': [{'answer': '1998-2022', 'is_correct': True}, {'answer': '2000-2022', 'is_correct': False}, {'answer': '1995-2020', 'is_correct': False}, {'answer': '2010-2022', 'is_correct': False}]}, {'question': 'Which seasonal forecast influenced the choice of percentile range for MAM 2024?', 'options': [{'answer': 'ECMWF prediction of above average precipitation', 'is_correct': True}, {'answer': 'NOAA prediction of normal precipitation', 'is_correct': False}, {'answer': 'WMO forecast of below average rainfall', 'is_correct': False}, {'answer': 'Local meteorological department forecast', 'is_correct': False}]}], [{'question': 'What threshold was used to reclassify flood fraction composites to binary in the analysis?', 'options': [{'answer': '20 percent', 'is_correct': True}, {'answer': '5 percent', 'is_correct': False}, {'answer': '10 percent', 'is_correct': False}, {'answer': '15 percent', 'is_correct': False}]}, {'question': 'Which percentile range was used for the OND 2024 flood exposure predictions?', 'options': [{'answer': '25-75th percentile', 'is_correct': True}, {'answer': '50-95th percentile', 'is_correct': False}, {'answer': '10-50th percentile', 'is_correct': False}, {'answer': '75-95th percentile', 'is_correct': False}]}, {'question': 'What population dataset was used in the flood exposure analysis?', 'options': [{'answer': 'WorldPop (2020 UN Adjusted)', 'is_correct': True}, {'answer': 'WorldPop (2022 UN Adjusted)', 'is_correct': False}, {'answer': 'WorldPop (2019 UN Adjusted)', 'is_correct': False}, {'answer': 'WorldPop (2021 UN Adjusted)', 'is_correct': False}]}, {'question': 'What time period of FloodScan data was analyzed in the study?', 'options': [{'answer': '1998-2022', 'is_correct': True}, {'answer': '2000-2022', 'is_correct': False}, {'answer': '1995-2022', 'is_correct': False}, {'answer': '2010-2022', 'is_correct': False}]}, {'question': "Which percentile range was used for MAM 2024 flood exposure predictions due to ECMWF's forecast of above-average precipitation?", 'options': [{'answer': '50-95th percentile', 'is_correct': True}, {'answer': '25-75th percentile', 'is_correct': False}, {'answer': '75-100th percentile', 'is_correct': False}, {'answer': '40-85th percentile', 'is_correct': False}]}], [{'question': 'What population dataset was used in combination with FloodScan data for the flood exposure analysis?', 'options': [{'answer': 'WorldPop (2020 UN Adjusted)', 'is_correct': True}, {'answer': 'UNFPA 2024 Dataset', 'is_correct': False}, {'answer': 'UN OCHA 2023 Population Data', 'is_correct': False}, {'answer': 'Somalia Census 2022', 'is_correct': False}]}, {'question': 'What threshold was used to reclassify the flood fraction composites to binary?', 'options': [{'answer': '20 percent', 'is_correct': True}, {'answer': '10 percent', 'is_correct': False}, {'answer': '5 percent', 'is_correct': False}, {'answer': '15 percent', 'is_correct': False}]}, {'question': 'Which percentile range was used for the OND 2024 flood exposure predictions?', 'options': [{'answer': '25-75th percentile', 'is_correct': True}, {'answer': '50-95th percentile', 'is_correct': False}, {'answer': '10-90th percentile', 'is_correct': False}, {'answer': '30-70th percentile', 'is_correct': False}]}, {'question': 'What time period of historical FloodScan data was analyzed?', 'options': [{'answer': '1998-2022', 'is_correct': True}, {'answer': '2000-2023', 'is_correct': False}, {'answer': '1995-2020', 'is_correct': False}, {'answer': '2010-2022', 'is_correct': False}]}, {'question': 'Which seasonal forecast influenced the choice of percentile range for MAM 2024?', 'options': [{'answer': 'ECMWF prediction of above average precipitation', 'is_correct': True}, {'answer': 'NOAA prediction of normal rainfall', 'is_correct': False}, {'answer': 'WMO forecast of below average precipitation', 'is_correct': False}, {'answer': 'Local meteorological department prediction', 'is_correct': False}]}], [{'question': 'What threshold was used to reclassify flood fraction composites to binary in the analysis?', 'options': [{'answer': '20 percent', 'is_correct': True}, {'answer': '5 percent', 'is_correct': False}, {'answer': '10 percent', 'is_correct': False}, {'answer': '15 percent', 'is_correct': False}]}, {'question': 'Which percentile range was used for the OND 2024 flood exposure estimates?', 'options': [{'answer': '25-75th percentile', 'is_correct': True}, {'answer': '50-95th percentile', 'is_correct': False}, {'answer': '10-90th percentile', 'is_correct': False}, {'answer': '5-95th percentile', 'is_correct': False}]}, {'question': 'What population dataset was used in the flood exposure analysis?', 'options': [{'answer': 'WorldPop (2020 UN Adjusted)', 'is_correct': True}, {'answer': 'WorldPop (2022 UN Adjusted)', 'is_correct': False}, {'answer': 'UN OCHA Population Data 2024', 'is_correct': False}, {'answer': 'UNFPA Census Data 2020', 'is_correct': False}]}, {'question': 'What was the time period of the FloodScan data used in the analysis?', 'options': [{'answer': '1998-2022', 'is_correct': True}, {'answer': '2000-2024', 'is_correct': False}, {'answer': '1995-2020', 'is_correct': False}, {'answer': '2010-2022', 'is_correct': False}]}, {'question': 'Which seasonal forecast influenced the choice of percentile range for MAM 2024?', 'options': [{'answer': 'ECMWF prediction of above average precipitation', 'is_correct': True}, {'answer': 'NOAA prediction of normal rainfall', 'is_correct': False}, {'answer': 'WMO forecast of below average precipitation', 'is_correct': False}, {'answer': 'Local meteorological department forecast', 'is_correct': False}]}], [{'question': 'What population dataset was used in combination with FloodScan data for the flood exposure analysis?', 'options': [{'answer': 'WorldPop 2020 UN Adjusted', 'is_correct': True}, {'answer': 'UNFPA 2024 Dataset', 'is_correct': False}, {'answer': 'UN OCHA 2023 Population Data', 'is_correct': False}, {'answer': 'Somalia Census 2022', 'is_correct': False}]}, {'question': 'What threshold was used to reclassify the flood fraction composites to binary?', 'options': [{'answer': '20 percent', 'is_correct': True}, {'answer': '10 percent', 'is_correct': False}, {'answer': '5 percent', 'is_correct': False}, {'answer': '15 percent', 'is_correct': False}]}, {'question': 'Which percentile range was used for the OND 2024 flood exposure predictions?', 'options': [{'answer': '25-75th percentile', 'is_correct': True}, {'answer': '50-95th percentile', 'is_correct': False}, {'answer': '10-90th percentile', 'is_correct': False}, {'answer': '30-70th percentile', 'is_correct': False}]}, {'question': 'What time period of historical FloodScan data was analyzed?', 'options': [{'answer': '1998-2022', 'is_correct': True}, {'answer': '2000-2023', 'is_correct': False}, {'answer': '1995-2020', 'is_correct': False}, {'answer': '2010-2022', 'is_correct': False}]}, {'question': 'Which seasonal forecast influenced the choice of percentile range for MAM 2024?', 'options': [{'answer': 'ECMWF prediction of above average precipitation', 'is_correct': True}, {'answer': 'NOAA prediction of normal rainfall', 'is_correct': False}, {'answer': 'WMO forecast of below average precipitation', 'is_correct': False}, {'answer': 'Local meteorological department forecast', 'is_correct': False}]}], [{'question': 'What threshold was used to reclassify flood fraction composites to binary in the analysis?', 'options': [{'answer': '20 percent', 'is_correct': True}, {'answer': '5 percent', 'is_correct': False}, {'answer': '10 percent', 'is_correct': False}, {'answer': '15 percent', 'is_correct': False}]}, {'question': 'Which percentile range was used for the OND 2024 flood exposure estimates?', 'options': [{'answer': '25-75th percentile', 'is_correct': True}, {'answer': '50-95th percentile', 'is_correct': False}, {'answer': '10-50th percentile', 'is_correct': False}, {'answer': '75-95th percentile', 'is_correct': False}]}, {'question': 'What population dataset was used in the flood exposure analysis?', 'options': [{'answer': 'WorldPop (2020 UN Adjusted)', 'is_correct': True}, {'answer': 'WorldPop (2022 UN Adjusted)', 'is_correct': False}, {'answer': 'WorldPop (2019 UN Adjusted)', 'is_correct': False}, {'answer': 'WorldPop (2021 UN Adjusted)', 'is_correct': False}]}, {'question': 'What years of FloodScan data were included in the analysis?', 'options': [{'answer': '1998-2022', 'is_correct': True}, {'answer': '2000-2022', 'is_correct': False}, {'answer': '1995-2022', 'is_correct': False}, {'answer': '2010-2022', 'is_correct': False}]}, {'question': 'Which seasonal forecast influenced the choice of percentile range for MAM 2024?', 'options': [{'answer': 'ECMWF prediction of above average precipitation', 'is_correct': True}, {'answer': 'ECMWF prediction of below average precipitation', 'is_correct': False}, {'answer': 'Local weather station data', 'is_correct': False}, {'answer': 'Historical flooding patterns', 'is_correct': False}]}], [{'question': 'What threshold was used to reclassify flood fraction composites to binary in the analysis?', 'options': [{'answer': '20 percent', 'is_correct': True}, {'answer': '5 percent', 'is_correct': False}, {'answer': '10 percent', 'is_correct': False}, {'answer': '15 percent', 'is_correct': False}]}, {'question': 'Which percentile range was used for the OND 2024 flood exposure estimates?', 'options': [{'answer': '25-75th percentile', 'is_correct': True}, {'answer': '50-95th percentile', 'is_correct': False}, {'answer': '10-90th percentile', 'is_correct': False}, {'answer': '15-85th percentile', 'is_correct': False}]}, {'question': 'What population dataset was used in the analysis?', 'options': [{'answer': 'WorldPop (2020 UN Adjusted)', 'is_correct': True}, {'answer': 'WorldPop (2022 UN Adjusted)', 'is_correct': False}, {'answer': 'Somalia Census Data 2020', 'is_correct': False}, {'answer': 'UNFPA Population Database 2022', 'is_correct': False}]}, {'question': 'What time period of FloodScan data was analyzed?', 'options': [{'answer': '1998-2022', 'is_correct': True}, {'answer': '2000-2022', 'is_correct': False}, {'answer': '1995-2022', 'is_correct': False}, {'answer': '2010-2022', 'is_correct': False}]}, {'question': 'Which seasonal periods were analyzed in the flood exposure methodology?', 'options': [{'answer': 'March-April-May (MAM) and October-November-December (OND)', 'is_correct': True}, {'answer': 'January-February-March (JFM) and July-August-September (JAS)', 'is_correct': False}, {'answer': 'April-May-June (AMJ) and September-October-November (SON)', 'is_correct': False}, {'answer': 'February-March-April (FMA) and August-September-October (ASO)', 'is_correct': False}]}]]
    
    #values = evaluator.collect_all_values(data)
    #print(values)
    
    json1 = [{'question': 'What data was used to calculate the number of people potentially exposed to flooding in Somalia for the 2024 HNRP?', 'options': [{'answer': 'Daily FloodScan (1998-2022) & WorldPop (2020 UN Adjusted) raster data', 'is_correct': True}, {'answer': 'ECMWF seasonal forecast data only', 'is_correct': False}, {'answer': 'UNFPA Methodology data only', 'is_correct': False}, {'answer': 'Somalia ICCG and HCT data only', 'is_correct': False}]}, {'question': 'What threshold was used to reclassify the FloodScan daily flood fraction Standard Flood Exposure Depiction (SFED) to binary?', 'options': [{'answer': '10 percent flood fraction threshold', 'is_correct': False}, {'answer': '20 percent flood fraction threshold', 'is_correct': True}, {'answer': '30 percent flood fraction threshold', 'is_correct': False}, {'answer': '40 percent flood fraction threshold', 'is_correct': False}]}, {'question': 'How were the yearly seasonal flood exposure rasters aggregated to obtain the estimated population exposure per district?', 'options': [{'answer': 'Via zonal statistics (mean)', 'is_correct': False}, {'answer': 'Via zonal statistics (sum)', 'is_correct': True}, {'answer': 'Via zonal statistics (median)', 'is_correct': False}, {'answer': 'Via zonal statistics (mode)', 'is_correct': False}]}, {'question': 'What percentiles were used to estimate the range of population exposed for the MAM 2024 season?', 'options': [{'answer': '25th-50th percentile levels', 'is_correct': False}, {'answer': '50th-95th percentile levels', 'is_correct': True}, {'answer': '25th-75th percentile levels', 'is_correct': False}, {'answer': '10th-90th percentile levels', 'is_correct': False}]}, {'question': 'Who should be contacted for questions or feedback on the Somalia Flood Exposure Methodology Note?', 'options': [{'answer': 'Leonardo Milano, Team Lead for Data Science at leonardo.milano@un.org', 'is_correct': True}, {'answer': 'Somalia ICCG and HCT', 'is_correct': False}, {'answer': 'UNFPA Methodology team', 'is_correct': False}, {'answer': 'ECMWF seasonal forecast team', 'is_correct': False}]}]

    json2 = [{'question': 'What data sources were used to analyze flood conditions in Somalia?', 'options': [{'answer': 'Daily FloodScan (1998-2022) and WorldPop (2020 UN Adjusted) raster data', 'is_correct': True}, {'answer': 'ECMWF seasonal forecast data only', 'is_correct': False}, {'answer': 'UN OCHA population estimates', 'is_correct': False}, {'answer': 'Satellite imagery from 2024', 'is_correct': False}]}, {'question': 'How were the flood fraction composites processed?', 'options': [{'answer': 'Reclassified to binary using a 20 percent flood fraction threshold and masked to remove flood fraction values < 0.05 percent', 'is_correct': True}, {'answer': 'Completely removed from the analysis', 'is_correct': False}, {'answer': 'Converted directly to population exposure estimates', 'is_correct': False}, {'answer': 'Averaged across all years without further processing', 'is_correct': False}]}, {'question': 'What seasons were analyzed for flood exposure in Somalia?', 'options': [{'answer': 'March-April-May (MAM) and October-November-December (OND) seasons', 'is_correct': True}, {'answer': 'January-February and July-August', 'is_correct': False}, {'answer': 'Only the MAM season', 'is_correct': False}, {'answer': 'Winter and summer seasons', 'is_correct': False}]}, {'question': 'How were the population exposure ranges estimated?', 'options': [{'answer': 'Using percentiles for MAM (50th-95th) and OND (25th-75th) seasons', 'is_correct': True}, {'answer': 'Using exact population count predictions', 'is_correct': False}, {'answer': 'Based solely on ECMWF seasonal forecast', 'is_correct': False}, {'answer': 'Using a fixed percentage across all districts', 'is_correct': False}]}, {'question': 'What was the final step in calculating population flood exposure?', 'options': [{'answer': 'Applying the percent exposure to the 2024 population data set and aggregating to administrative level 1', 'is_correct': True}, {'answer': 'Directly using the FloodScan data', 'is_correct': False}, {'answer': 'Multiplying by a fixed population growth factor', 'is_correct': False}, {'answer': 'Estimating exposure based on historical data only', 'is_correct': False}]}]
    
    json3 = [{'question': 'What data sources were used to analyze flood conditions in Somalia?', 'options': [{'answer': 'Daily FloodScan (1998-2022) and WorldPop (2020 UN Adjusted) raster data', 'is_correct': True}, {'answer': 'ECMWF seasonal forecast data only', 'is_correct': False}, {'answer': 'UN OCHA population estimates', 'is_correct': False}, {'answer': 'UNFPA demographic data', 'is_correct': False}]}, {'question': 'How were the flood fraction composites processed?', 'options': [{'answer': 'Reclassified to binary using a 20 percent flood fraction threshold and masked to remove values < 0.05 percent', 'is_correct': True}, {'answer': 'Completely removed from the analysis', 'is_correct': False}, {'answer': 'Converted directly to population exposure estimates', 'is_correct': False}, {'answer': 'Aggregated without any processing', 'is_correct': False}]}, {'question': 'What seasonal periods were analyzed for flood exposure?', 'options': [{'answer': 'March-April-May (MAM) and October-November-December (OND) seasons', 'is_correct': True}, {'answer': 'January-February and June-July', 'is_correct': False}, {'answer': 'Only MAM season', 'is_correct': False}, {'answer': 'Only OND season', 'is_correct': False}]}, {'question': 'How were the flood exposure ranges estimated?', 'options': [{'answer': 'Using percentiles for MAM (50th-95th) and OND (25th-75th) seasons', 'is_correct': True}, {'answer': 'Using fixed percentage across all districts', 'is_correct': False}, {'answer': 'Based solely on ECMWF seasonal forecast', 'is_correct': False}, {'answer': 'Randomly generated estimates', 'is_correct': False}]}, {'question': 'What was the final step in calculating population flood exposure?', 'options': [{'answer': 'Applying percent exposure to the 2024 population dataset and aggregating to administrative level 1', 'is_correct': True}, {'answer': 'Directly using historical flood data', 'is_correct': False}, {'answer': 'Multiplying by a fixed population coefficient', 'is_correct': False}, {'answer': 'Removing all low-risk districts', 'is_correct': False}]}]
    
    json1 = [{"name": "John", "age": 30, "city": "NYC"}]
    json2 = [{"name": "wang", "age": 30, "city": "LA"}]
    json3 = [{"name": "Jane", "age": 10, "city": "NYC"}]
    
    
    result_semantic1 = evaluator.calculate_similarity_with_deepdiff(json1, json2)
    result_semantic2 = evaluator.calculate_similarity_with_deepdiff(json1, json3)
    print(f"deepdiff->result_semantic1: {result_semantic1}")
    print(f"deepdiff->result_semantic1: {result_semantic2}")
    
    t1 = {'a': 1, 'b': 2}
    t2 = {'a': 1, 'b': 3}

    diff1 = DeepDiff(json1, json2, ignore_order=True, cache_size=5000, get_deep_distance=True)
    diff2 = DeepDiff(json1, json3, ignore_order=True, cache_size=5000, get_deep_distance=True)
    print(diff1)
    print(diff2)
    