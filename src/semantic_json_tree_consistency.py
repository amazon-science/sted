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

from deepdiff import DeepDiff
import concurrent.futures

import itertools
import math

# Try relative imports first (when used as package)
from .json_tree_node import JsonNode
from .similarity_cache import StringSimilarityCache
from .utils import collect_all_values, getEmbeddings, create_bedrock_client, count_json_elements
from .gnn import compare_json
from .bedrock_utils import build_message, get_json, inference_with_converse_api
from botocore.config import Config

# System prompt for LLM judge to calculate similarity score between two structured data
SYSTEM_PROMPT_JUDGE = """You are an expert evaluator tasked with assessing the similarity between two structured JSON outputs. Your goal is to provide a similarity score that aligns with human judgment.

## Evaluation Criteria:

1. **Semantic Equivalence** (40%): Do the outputs convey the same meaning, even if expressed differently?
   - Consider synonyms, paraphrases, and equivalent expressions
   - Account for different levels of detail that maintain core meaning
   - Recognize when different structures represent the same information

2. **Structural Consistency** (30%): How well do the organizational patterns match?
   - Field names and their semantic relationships
   - Hierarchical organization and nesting patterns
   - Data types and their appropriateness for the content

3. **Content Accuracy** (20%): How accurate are the specific values and details?
   - Factual correctness of information
   - Precision of numerical values
   - Completeness of required information

4. **Functional Equivalence** (10%): Would these outputs serve the same purpose in practice?
   - Usability for downstream applications
   - Preservation of key relationships and dependencies

## Scoring Guidelines:

- **0.9-1.0**: Essentially identical or semantically equivalent with minor stylistic differences
- **0.7-0.8**: Very similar with some differences in detail or structure that don't affect core meaning
- **0.5-0.6**: Moderately similar with notable differences but shared core concepts
- **0.3-0.4**: Some similarities but significant differences in structure or content
- **0.1-0.2**: Minimal similarity with only basic shared elements
- **0.0**: Completely different or unrelated

## Instructions:

1. Analyze both JSON structures carefully
2. Consider the context and likely use case
3. Focus on semantic meaning over exact textual matches
4. Provide a single similarity score between 0.0 and 1.0
5. Be consistent with human judgment patterns

Generate result using calculate_similarity_score."""

judge_schema = [
    {
        "toolSpec": {
            "name": "calculate_similarity_score",
            "description": "Calculate the similarity score between two structured JSON outputs",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "overall_score": {
                            "type": "number",
                            "description": "The similarity score between two structured JSON outputs, ranging from 0.0 to 1.0"
                        },
                        "structural_score": {
                            "type": "number",
                            "description": "The structural consistency score, ranging from 0.0 to 1.0"
                        },
                        "content_score": {
                            "type": "number",
                            "description": "The semantic equivalence score for values in json outputs, ranging from 0.0 to 1.0"
                        }
                    },
                    "required": ["overall_score", "structural_score", "content_score"]
                }
            }
        }
    }
]

# Define default configuration inline to avoid import issues
def _get_default_type_change_costs() -> Dict[Tuple[str, str], float]:
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
    costs[("string", "number")] = costs[("number", "string")] = 0.1
    costs[("boolean", "string")] = costs[("string", "boolean")] = 0.1
    costs[("number", "boolean")] = costs[("boolean", "number")] = 0.1
    costs[("null", "string")] = costs[("string", "null")] = 0.1
    costs[("null", "number")] = costs[("number", "null")] = 0.1
    
    # Higher costs for structure changes
    costs[("object", "array")] = costs[("array", "object")] = 0.5
    costs[("object", "string")] = costs[("string", "object")] = 0.5
    costs[("object", "number")] = costs[("number", "object")] = 0.5
    costs[("array", "string")] = costs[("string", "array")] = 0.5
    costs[("array", "number")] = costs[("number", "array")] = 1
    
    return costs


from transformers import logging
logging.set_verbosity_error()

class SemanticJsonTreeConsistencyEvaluator:
    """Evaluator for JSON structural consistency using Tree Edit Distance with semantic similarity."""
    
    def __init__(self, 
                 path_weight_decay: float = 1.0,
                 type_change_cost: Dict[Tuple[str, str], float] = None,
                 required_fields: Set[str] = None,
                 model_id: str = 'all-MiniLM-L6-v2',
                 chunk_size: int = 300,
                 chunk_overlap: int = 50,
                 region_name: str = "us-west-2",
                 weights: Dict[str, float] = {"type": 0.1, "value": 0.8, "key": 0.1},
                 sort_keys: bool = True,
                 sort_arrays: bool = True
        ):
        """
        Initialize the evaluator with semantic capabilities.
        
        Args:
            path_weight_decay: Weight decay factor for deeper paths (0-1)
            type_change_cost: Custom costs for type changes
            required_fields: Set of required field paths
            model_id: Name of the sentence transformer model or Bedrock model ID
            chunk_size: Size of chunks for text splitting
            chunk_overlap: Overlap between chunks
        """
        self.path_weight_decay = path_weight_decay
        self.type_change_cost = type_change_cost or _get_default_type_change_costs()
        self.required_fields = required_fields or set()
        
        # Initialize embedding model if available
        self.embedding_model = None
        self.bedrock_client = None
        self.llm_client = None
        self.model_id = model_id
        
        boto_config = Config(
            retries={
                'max_attempts': 10,
                'mode': 'adaptive'
            },
            max_pool_connections=50  # Increase connection pool size
        )
        
        if self.model_id in ["amazon.titan-embed-text-v1", "amazon.titan-embed-text-v2:0", "cohere.embed-multilingual-v3"]:
            self.bedrock_client = create_bedrock_client(region_name=region_name, config=boto_config)
        else:
            self.embedding_model = SentenceTransformer(self.model_id)
            # Warm up the model
            self.embedding_model.encode(["test"], show_progress_bar=False)
        
        # Initialize LLM client for similarity evaluation (using same Bedrock client)
        self.llm_client = self.bedrock_client if self.bedrock_client else create_bedrock_client(region_name=region_name, config=boto_config)
        
        # Cache for embeddings
        self._cache = StringSimilarityCache()
        
        # Text splitting configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.calculate_similarity_method = {
            "ted": self.calculate_tree_edit_distance,
            "sted": self.calculate_tree_edit_distance_opt,
            "bertscore": self.calculate_bertscore,
            "deepdiff": self.calculate_similarity_with_deepdiff,
            "deepdiff_opt": self.calculate_similarity_with_deepdiff_opt,
            "gnn": self.calculate_gnn_similarity,
            "llm_judge": self.calculate_similarity_with_llm
        }
        
        self.weights = weights
        
        self.sort_keys = sort_keys
        self.sort_arrays = sort_arrays
        
        self.batch_size_bertscore = 2000
        
        self.judge_model_id = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    
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
        similarity = np.clip(similarity, -1.0, 1.0)
        
        # Scale from [-1, 1] to [0, 1]
        similarity = (similarity + 1) / 2.0
        
        return float(similarity)

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
    
    def _compare_strings(self, str1: str, str2: str, method="cosine") -> float:
        """Compare two strings with optional semantic similarity and chunking for long text."""
        # Quick equality check
        
        if str1 == str2:
            return 1.0
        
        # Check cache first
        cached_sim = self._cache.get(str1, str2)
        if cached_sim is not None:
            return cached_sim
        
        if len(str1) < self.chunk_size and len(str2) < self.chunk_size:
            if method == "bertscore":
                P, R, F1 = bert_score([str1], [str2], lang="en")
                sim = float(F1.item())
            else:
                sim = self._calculate_semantic_similarity(str1, str2)
            self._cache.set(str1, str2, sim)
            return sim
        else:
            chunks1 = self._split_natural_text(str1)
            chunks2 = self._split_natural_text(str2)
            
            return 1 - self._compare_arrays_unordered(chunks1, chunks2, "str1_chunks", "str2_chunks")
        
    def _split_natural_text(self, text: str, coding_language: str=None) -> List[str]:
        """Split natural language text into sentences or paragraphs using LangChain if available and enabled."""
        # For short text, don't split at all
        if len(text) < self.chunk_size:
            return [text]
        
        if coding_language:
            #ToDo: add support for langchain
            text_splitter = None
            pass
        else:
            # Create a text splitter that tries to create semantically meaningful chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size - self.chunk_overlap,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ";", ":", " "]  # Try these separators in order
            )
        
        try:
            # Split the text
            return text_splitter.split_text(text)
        except Exception as e:
            warnings.warn(f"Failed to split text: {e}\n{text}")
            return [text]
    
    def _compare_numbers(self, num1: float, num2: float) -> float:
        """Compare two numbers with tolerance."""
        return 0 if num1 != num2 else 1
            
    def update_cost(self, node1: JsonNode, node2: JsonNode, structural_weight=0.5) -> float:
        """
        Calculate the cost of updating a node.
        
        Args:
            node1: The source node
            node2: The target node
            
        Returns:
            The cost of update
        """
        structural_cost = self.structural_update_cost(node1, node2)
        content_cost = self.content_update_cost(node1, node2)
        
        # Combined cost with weighting
        return structural_weight * structural_cost + (1 - structural_weight) * content_cost
    
    def _calculate_structural_similarity(self, node1: JsonNode, node2: JsonNode) -> float:
        """Calculate structural similarity: paths and field names only"""        
        path1 = re.sub(r'\[\d+\]', '[*]', node1.path)
        path2 = re.sub(r'\[\d+\]', '[*]', node2.path)
        path_match = path1 == path2
        
        key1 = node1.label.split('.')[-1] if '.' in node1.label else node1.label
        key2 = node2.label.split('.')[-1] if '.' in node2.label else node2.label
        key1 = re.sub(r'\[\d+\]', '', key1)
        key2 = re.sub(r'\[\d+\]', '', key2)
                
        # replace all special characters such as -, _, % with space
        key1 = re.sub(r'[^a-zA-Z0-9]', ' ', key1)
        key2 = re.sub(r'[^a-zA-Z0-9]', ' ', key2)
                
        field_similarity = self._calculate_semantic_similarity(key1, key2)
        structural_similarity = (field_similarity + (1 if path_match else 0)) / 2
        
        return structural_similarity
    
    def _calculate_content_similarity(self, node1: JsonNode, node2: JsonNode, structural_sim_threshold=0.3) -> float:
        """Calculate content similarity: includes type changes and value differences"""
        
        structural_sim = self._calculate_structural_similarity(node1, node2)
                
        if not node1.children and not node2.children and structural_sim > structural_sim_threshold:
            # Same type comparisons
            if node1.node_type == node2.node_type:
                if node1.node_type == "string":                    
                    return self._compare_strings(str(node1.value), str(node2.value))                
                elif node1.node_type == "number":
                    return self._compare_numbers(float(node1.value), float(node2.value))
                elif node1.node_type == "array":
                    return 1 - self._compare_arrays_unordered(list(node1.value), list(node2.value), node1.label, node2.label, variation_type="content")
                elif node1.node_type == "object":
                    return self._calculate_content_similarity(node1.value, node2.value, structural_sim_threshold=structural_sim_threshold)
                else:
                    return 1.0
            else:
                val1_str = str(node1.value)
                val2_str = str(node2.value)
                                
                # Use semantic similarity on string representations
                semantic_sim = self._calculate_semantic_similarity(val1_str, val2_str)
                
                # Apply small type penalty
                type_cost = self.type_change_cost.get((node1.node_type, node2.node_type), 1.0)
                content_similarity = max(0.0, semantic_sim - type_cost)
                
                return content_similarity
        
        return 0.0
        
    def structural_update_cost(self, node1: JsonNode, node2: JsonNode) -> float:
        """Calculate structural update cost only"""
        if (node1.label == node2.label and 
            node1.node_type == node2.node_type and 
            len(node1.children) == len(node2.children)):
            return 0.0  # Structurally identical
        
        structural_similarity = self._calculate_structural_similarity(node1, node2)
        cost = 1 - structural_similarity
        
        # Apply path weighting
        path_weight1 = self.calculate_path_weight(node1.path)
        path_weight2 = self.calculate_path_weight(node2.path)
        avg_path_weight = (path_weight1 + path_weight2) / 2.0
        
        return cost * avg_path_weight
    
    def content_update_cost(self, node1: JsonNode, node2: JsonNode) -> float:
        """Calculate content update cost only"""
        if (node1.node_type == node2.node_type and 
            node1.value == node2.value):
            return 0.0  # Content identical
        
        content_similarity = self._calculate_content_similarity(node1, node2)
        cost = 1 - content_similarity
        
        # Apply path weighting
        path_weight1 = self.calculate_path_weight(node1.path)
        path_weight2 = self.calculate_path_weight(node2.path)
        avg_path_weight = (path_weight1 + path_weight2) / 2.0
        
        return cost * avg_path_weight
    
    def _compare_arrays_unordered(self, arr1: List[Any], arr2: List[Any], arr1_label: str, arr2_label: str, variation_type="content") -> float:
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
                    tree1 = JsonNode.from_dict(item1, f"{arr1_label}", sort_arrays=self.sort_arrays, sort_keys=self.sort_keys)
                    tree2 = JsonNode.from_dict(item2, f"{arr2_label}", sort_arrays=self.sort_arrays, sort_keys=self.sort_keys)
                    cost_matrix[i, j] = self._calculate_optimal_matching_cost(tree1, tree2, variation_type=variation_type)
                elif isinstance(item1, list):
                    # Recursive array comparison
                    cost_matrix[i, j] = self._compare_arrays_unordered(item1, item2, arr1_label, arr2_label, variation_type=variation_type)
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
        total_cost += max(max(0, unmatched_from_arr1), max(0, unmatched_from_arr2))
        
        # Normalize by the maximum array length
        normalized_cost = total_cost / max_len if max_len > 0 else 0
                
        return min(normalized_cost, 1)
    
    def _calculate_optimal_matching_cost(
        self, tree1: JsonNode, tree2: JsonNode, variation_type: str = "combined"
    ) -> float:
        """
        Calculate optimal matching cost between two trees using Hungarian algorithm.
        
        Args:
            tree1: First tree
            tree2: Second tree  
            variation_type: "structural", "content", or "combined"
        """
        # Base case: both are leaf nodes
        if not tree1.children and not tree2.children:
            if variation_type == "structural":
                return self.structural_update_cost(tree1, tree2)
            elif variation_type == "content":
                return self.content_update_cost(tree1, tree2)
            else:  # combined
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
            if variation_type == "structural":
                return self.structural_update_cost(tree1, tree2)
            elif variation_type == "content":
                return self.content_update_cost(tree1, tree2)
            else:
                return self.update_cost(tree1, tree2)

        # Create cost matrix for Hungarian algorithm
        n1, n2 = len(children1), len(children2)
        max_size = max(n1, n2)
        cost_matrix = [[float('inf')] * max_size for _ in range(max_size)]

        # Fill cost matrix
        for i in range(n1):
            for j in range(n2):
                # Cost of matching child i with child j
                cost_matrix[i][j] = self._calculate_optimal_matching_cost(
                    children1[i], children2[j], variation_type
                )

        # Add deletion costs for unmatched nodes in tree1
        for i in range(n1):
            for j in range(n2, max_size):
                cost_matrix[i][j] = self.delete_cost(children1[i])

        # Add insertion costs for unmatched nodes in tree2
        for i in range(n1, max_size):
            for j in range(n2):
                cost_matrix[i][j] = self.insert_cost(children2[j])

        # Solve assignment problem using Hungarian algorithm
        from scipy.optimize import linear_sum_assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        total_cost = sum(cost_matrix[i][j] for i, j in zip(row_indices, col_indices))
        
        # Calculate actual matched pairs
        matched_pairs = sum(1 for i, j in zip(row_indices, col_indices) if i < n1 and j < n2)
        unmatched_total = (n1 - matched_pairs) + (n2 - matched_pairs)
        
        # Add size penalty based on unmatched elements
        size_penalty = min(unmatched_total * 0.1, max_size * 0.3)
        
        total_cost += size_penalty
                
        normalized_cost = total_cost / len(row_indices)
        normalized_cost = min(normalized_cost, 1.0)
        
        #return total_cost if total_cost > 0 else 0
        return normalized_cost
    
    def calculate_tree_edit_distance_opt(self, json1: Dict[str, Any], json2: Dict[str, Any], variation_type: str = "combined") -> float:
        return self.calculate_tree_edit_distance(json1, json2, original_zss=False, variation_type=variation_type)
    
    def calculate_tree_edit_distance(self, json1: Dict[str, Any], json2: Dict[str, Any], original_zss=True, variation_type: str = "combined") -> float:
        """
        Calculate tree edit distance between two JSON objects.
        
        Args:
            json1: First JSON object
            json2: Second JSON object
            original_zss: Whether to use original ZSS algorithm
            variation_type: "structural", "content", or "combined"
            
        Returns:
            similarity_score
        """
        json1 = {"root": json1} if isinstance(json1, dict) else json1
        json2 = {"root": json2} if isinstance(json2, dict) else json2
        
        # Convert JSONs to trees
        tree1 = JsonNode.from_dict(json1, sort_arrays=self.sort_arrays, sort_keys=self.sort_keys)
        tree2 = JsonNode.from_dict(json2, sort_arrays=self.sort_arrays, sort_keys=self.sort_keys)
        
        if original_zss:
            try:
                # For ZSS, use appropriate cost function based on variation_type
                """
                if variation_type == "structural":
                    update_cost_func = self.structural_update_cost
                elif variation_type == "content":
                    update_cost_func = self.content_update_cost
                else:
                    update_cost_func = self.update_cost
                
                distance = zss.distance(
                    tree1, tree2,
                    get_children=lambda x: x.get_children(),
                    insert_cost=self.insert_cost,
                    remove_cost=self.delete_cost,
                    update_cost=update_cost_func
                )
                """
                distance = zss.simple_distance(tree1, tree2, get_children=lambda x: x.get_children())
                max_node = (tree1.count_nodes(), tree2.count_nodes())
                return 1 - distance/max_node
                
            except TypeError as e:
                raise TypeError(
                    f"Failed to calculate tree distance. Ensure zss is properly installed "
                    f"and trees have compatible structure: {str(e)}"
                ) from e
        else:
            distance = self._calculate_optimal_matching_cost(tree1, tree2, variation_type)
                
        similarity = 1.0 - distance
        return similarity
    
    def calculate_bertscore(self, json1: Dict[str, Any], json2: Dict[str, Any], **kwargs) -> float:
        # Preprocess JSONs to make them order-invariant
        tree1 = JsonNode.from_dict(json1, sort_arrays=self.sort_arrays, sort_keys=self.sort_keys)
        tree2 = JsonNode.from_dict(json2, sort_arrays=self.sort_arrays, sort_keys=self.sort_keys)
        processed_json1 = tree1.reconstruct_json()
        processed_json2 = tree2.reconstruct_json()
        
        P, R, F1 = bert_score([str(processed_json1)], [str(processed_json2)], lang="en")
        return float(F1.item())
    
    def calculate_similarity_with_llm(self, json1: [Dict[str, Any], List], json2: [Dict[str, Any], List], 
                                     llm_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0", 
                                     max_retries: int = 3, **kwargs) -> float:
        """
        Calculate similarity between two JSON structures using an LLM judge.
        
        Args:
            json1: First JSON object or list
            json2: Second JSON object or list
            llm_model_id: The LLM model to use for evaluation
            max_retries: Maximum number of retry attempts
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            # Prepare the JSON structures for comparison
            json1_str = json.dumps(json1, indent=2, ensure_ascii=False, sort_keys=True)
            json2_str = json.dumps(json2, indent=2, ensure_ascii=False, sort_keys=True)
            
            # Create the evaluation prompt
            user_prompt = f"""Please evaluate the similarity between these two JSON structures:

JSON Structure 1:
```json
{json1_str}
```

JSON Structure 2:
```json
{json2_str}
```

Provide a similarity score between 0.0 and 1.0 based on semantic equivalence, structural consistency, content accuracy, and functional equivalence."""
            message = build_message(texts=[user_prompt])
            response = inference_with_converse_api(
                self.llm_client,
                model_id=llm_model_id,
                messages=[message],
                system_prompts=SYSTEM_PROMPT_JUDGE,
                max_tokens=2000,
                temperature=0.1,
                tools=judge_schema
            )
            
            return get_json(response, "calculate_similarity_score")
        except Exception as e:
            warnings.warn(f"Error in LLM similarity calculation: {str(e)}")
            # print error trace
            import traceback
            traceback.print_exc()
            return {}
        
    def calculate_gnn_similarity(self, json1: [Dict[str, Any], List], json2: [Dict[str, Any], List], **kwargs) -> float:
        """
        Calculate similarity using GNN approach from gnn.py.
        
        Args:
            json1: First JSON object or list
            json2: Second JSON object or list
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            if isinstance(json1, list):
                json1 = {"root": json1}
            if isinstance(json2, list):
                json2 = {"root": json2}
            
            # Check if we have the embedding model
            text_model = SentenceTransformer("all-MiniLM-L6-v2")
            
            # Use the compare_json function from gnn.py
            similarity = compare_json(json1, json2, text_model)
            
            return float(max(0.0, min(1.0, similarity)))
            
        except Exception as e:
            warnings.warn(f"Error in GNN similarity calculation: {str(e)}")
            return 0.0
    
    def batch_compute_llm_similarities(self, json_pairs: List[Tuple[Dict, Dict]], 
                                      llm_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0") -> Dict[Tuple[str, str], float]:
        """
        Batch compute LLM similarities for multiple JSON pairs.
        
        Args:
            json_pairs: List of (json1, json2) tuples to evaluate
            llm_model_id: The LLM model to use for evaluation
            
        Returns:
            Dictionary mapping JSON pair hashes to similarity scores
        """
        results = {}
        
        for json1, json2 in json_pairs:
            # Create a hash key for caching
            json1_str = json.dumps(json1, sort_keys=True)
            json2_str = json.dumps(json2, sort_keys=True)
            cache_key = (json1_str, json2_str)
            
            # Check cache first
            cached_score = self._cache.get(json1_str, json2_str)
            if cached_score is not None:
                results[cache_key] = cached_score
                continue
            
            # Calculate similarity using LLM
            try:
                similarity = self.calculate_similarity_with_llm(json1, json2, llm_model_id=llm_model_id)
                results[cache_key] = similarity
                
                # Cache the result
                self._cache.set(json1_str, json2_str, similarity)
                
            except Exception as e:
                warnings.warn(f"Failed to calculate LLM similarity for pair: {str(e)}")
                results[cache_key] = 0.0
        
        return results
    
    def calculate_similarity_with_deepdiff(self, json1: [Dict[str, Any], List], json2: [Dict[str, Any], List], **kwargs) -> float:
        diff = DeepDiff(json1, json2, ignore_order=True, cache_size=5000, get_deep_distance=True)
        return 1- diff['deep_distance']
    
    def calculate_similarity_with_deepdiff_opt(self, json1: [Dict[str, Any], List], json2: [Dict[str, Any], List], variation_type: str = "combined", structural_weight=0.5, **kwargs) -> float:
        """
        Calculate similarity using DeepDiff with enhanced value comparison.
        Uses semantic similarity for strings and proper comparison for numbers.
        
        Args:
            json1: First JSON object
            json2: Second JSON object
            variation_type: "structural", "content", or "combined"
            
        Returns:
            Similarity score between 0 and 1 (1 = identical, 0 = completely different)
        """
        try:
            # Use DeepDiff to find structural differences
            diff = DeepDiff(json1, json2, ignore_order=True, cache_size=5000)
            
            # If no differences found, return perfect similarity
            if not diff:
                return 1.0
            
            # Calculate similarity based on variation type
            if variation_type == "structural":
                return self._calculate_deepdiff_structural_only(diff, json1, json2)
            elif variation_type == "content":
                return self._calculate_deepdiff_content_only(diff, json1, json2)
            else:  # combined
                structural_sim = self._calculate_deepdiff_structural_only(diff, json1, json2)
                content_sim = self._calculate_deepdiff_content_only(diff, json1, json2)
                return structural_sim * structural_weight + (1 - structural_weight) * content_sim
                
        except Exception as e:
            warnings.warn(f"Error in DeepDiff calculation: {str(e)}")
            return 0.0
    
    def _calculate_deepdiff_structural_only(self, diff: dict, json1, json2) -> float:
        """Calculate structural similarity only (schema organization changes)"""
        structural_changes = 0
                
        # Count structural changes only (schema organization)
        if 'dictionary_item_added' in diff:
            structural_changes += len(diff['dictionary_item_added'])
        
        if 'dictionary_item_removed' in diff:
            structural_changes += len(diff['dictionary_item_removed'])
        
        if 'iterable_item_added' in diff:
            structural_changes += len(diff['iterable_item_added'])
        
        if 'iterable_item_removed' in diff:
            structural_changes += len(diff['iterable_item_removed'])
                
        # Estimate total structural elements
        total_elements = count_json_elements(json1) + count_json_elements(json2)
        if total_elements == 0:
            return 1.0
        
        # Calculate structural similarity
        structural_similarity = max(0.0, 1.0 - (structural_changes * 2) / total_elements)
        return structural_similarity
    
    def _calculate_deepdiff_content_only(self, diff: dict, json1, json2) -> float:
        """Calculate content similarity only (reuse original value processing logic)"""
        # First check structural similarity
        structural_sim = self._calculate_deepdiff_structural_only(diff, json1, json2)
        
        if structural_sim < 0.5:  # Same threshold as other methods
            return 0.0
        
        # Reuse original value processing logic
        total_similarity_score = 0.0
        total_comparisons = 0
        
        # Handle value changes with semantic comparison (original logic)
        if 'values_changed' in diff:
            for path, change in diff['values_changed'].items():
                old_value = change['old_value']
                new_value = change['new_value']
                
                # Use appropriate comparison method based on value types
                if isinstance(old_value, str) and isinstance(new_value, str):
                    value_similarity = self._compare_strings(old_value, new_value)
                elif isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                    value_similarity = self._compare_numbers(float(old_value), float(new_value))
                elif isinstance(old_value, bool) and isinstance(new_value, bool):
                    value_similarity = 1.0 if old_value == new_value else 0.0
                elif old_value is None or new_value is None:
                    value_similarity = 1.0 if old_value == new_value else 0.0
                elif isinstance(old_value, (list, dict)) and isinstance(new_value, (list, dict)):
                    value_similarity = self.calculate_similarity_with_deepdiff(old_value, new_value)
                else:
                    try:
                        value_similarity = self._compare_numbers(float(old_value), float(new_value))
                    except:
                        value_similarity = self._compare_strings(str(old_value), str(new_value))
                
                total_similarity_score += value_similarity
                total_comparisons += 1
        
        # Handle type changes as content changes (moved from structural)
        if 'type_changes' in diff:
            for path, change in diff['type_changes'].items():
                old_type = change['old_type'].__name__ if hasattr(change['old_type'], '__name__') else str(change['old_type'])
                new_type = change['new_type'].__name__ if hasattr(change['new_type'], '__name__') else str(change['new_type'])
                
                # Map Python types to our type system
                type_mapping = {
                    'str': 'string',
                    'int': 'number', 
                    'float': 'number',
                    'bool': 'boolean',
                    'NoneType': 'null',
                    'list': 'array',
                    'dict': 'object'
                }
                
                old_type_mapped = type_mapping.get(old_type, old_type)
                new_type_mapped = type_mapping.get(new_type, new_type)
                
                # Use type_change_cost to calculate type similarity
                type_cost = self.type_change_cost.get((old_type_mapped, new_type_mapped), 1.0)
                type_similarity = 1.0 - type_cost
                
                total_similarity_score += type_similarity
                total_comparisons += 1
        
        # Return content similarity
        if total_comparisons == 0:
            return 1.0  # No value or type changes
        
        return total_similarity_score / total_comparisons

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
    
    
    result_semantic1 = evaluator.calculate_tree_edit_distance(json1, json2)
    result_semantic2 = evaluator.calculate_tree_edit_distance(json1, json3)
    print(f"deepdiff->result_semantic1: {result_semantic1}")
    print(f"deepdiff->result_semantic1: {result_semantic2}")
    
    t1 = {'a': 1, 'b': 2}
    t2 = {'a': 1, 'b': 3}

    diff1 = DeepDiff(json1, json2, ignore_order=True, cache_size=5000, get_deep_distance=True)
    diff2 = DeepDiff(json1, json3, ignore_order=True, cache_size=5000, get_deep_distance=True)
    print(diff1)
    print(diff2)
    