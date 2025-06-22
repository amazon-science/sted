"""
JSON Structural Consistency Evaluation using Tree Edit Distance

This module provides functionality to evaluate the structural consistency of JSON outputs
using Tree Edit Distance algorithms. It converts JSON structures to trees and calculates
edit distances between them to measure structural similarity.
"""

import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set, Union
from collections import defaultdict
import datetime

# Try to import zss for Zhang-Shasha algorithm
try:
    import zss
    ZSS_AVAILABLE = True
except ImportError:
    ZSS_AVAILABLE = False
    print("Warning: zss package not found. Using custom tree edit distance implementation.")

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


class JsonTreeConsistencyEvaluator:
    """Evaluator for JSON structural consistency using Tree Edit Distance."""
    
    def __init__(self, 
                 schema_aware: bool = False, 
                 array_order_matters: bool = True,
                 path_weight_decay: float = 0.9,
                 type_change_cost: Dict[Tuple[str, str], float] = None,
                 required_fields: Set[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            schema_aware: Whether to use schema information if available
            array_order_matters: Whether the order of array elements matters
            path_weight_decay: Weight decay factor for deeper paths (0-1)
            type_change_cost: Custom costs for type changes
            required_fields: Set of required field paths
        """
        self.schema_aware = schema_aware
        self.array_order_matters = array_order_matters
        self.path_weight_decay = path_weight_decay
        self.type_change_cost = type_change_cost or self._default_type_change_costs()
        self.required_fields = required_fields or set()
        
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
            node = JsonNode(full_path, node_type="array")
            
            # Add children for each array item
            for i, item in enumerate(json_obj):
                child_path = f"{full_path}[{i}]" if full_path != "root" else f"[{i}]"
                child = self.json_to_tree(item, child_path, full_path)
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
    
    def are_nodes_equal(self, node1: JsonNode, node2: JsonNode) -> bool:
        """
        Determine if two nodes are considered equal.
        
        Args:
            node1: First node
            node2: Second node
            
        Returns:
            True if nodes are considered equal, False otherwise
        """
        # Basic equality: same type
        if node1.node_type != node2.node_type:
            return False
        
        # Compare full paths, normalizing array indices
        path1 = self._normalize_path(node1.path)
        path2 = self._normalize_path(node2.path)
        
        # If paths are different (after normalization), nodes are different
        if path1 != path2:
            return False
        
        # For leaf nodes, also compare values
        if not node1.children and not node2.children:
            # For primitive types, exact equality
            if node1.node_type in ["string", "number", "boolean", "null"]:
                return node1.value == node2.value
            # For empty objects or arrays, they're equal if they have the same type
            return True
        
        # For non-leaf nodes with same paths and types, they're considered equal
        # The actual differences in their children will be handled by the tree edit distance algorithm
        return True
        
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
        import re
        normalized = re.sub(r'\[\d+\]', '[*]', path)
        return normalized
    
    def update_cost(self, node1: JsonNode, node2: JsonNode) -> float:
        """
        Calculate the cost of updating a node.
        
        Args:
            node1: The source node
            node2: The target node
            
        Returns:
            The cost of update
        """
        # If nodes are identical according to our equality definition, no cost
        if self.are_nodes_equal(node1, node2):
            return 0.0
        
        # Base cost for type change
        type_cost = self.type_change_cost.get(
            (node1.node_type, node2.node_type), 
            1.0  # Default cost if not specified
        )
        
        # If same type but different values
        if node1.node_type == node2.node_type and node1.value != node2.value:
            # For primitive types, calculate value similarity
            if node1.node_type in ["string", "number", "boolean"]:
                if node1.node_type == "string":
                    # String similarity based on length of common substring
                    s1, s2 = str(node1.value), str(node2.value)
                    max_len = max(len(s1), len(s2))
                    if max_len == 0:
                        value_sim = 1.0
                    else:
                        # Use longest common subsequence for string similarity
                        lcs_len = self._longest_common_subsequence(s1, s2)
                        value_sim = lcs_len / max_len
                    value_cost = 1.0 - value_sim
                
                elif node1.node_type == "number":
                    # Number similarity based on relative difference
                    n1, n2 = float(node1.value), float(node2.value)
                    max_val = max(abs(n1), abs(n2))
                    if max_val == 0:
                        value_cost = 0.0  # Both are zero
                    else:
                        rel_diff = abs(n1 - n2) / max_val
                        value_cost = min(1.0, rel_diff)
                
                else:  # boolean
                    value_cost = 0.5  # Fixed cost for boolean change
            else:
                value_cost = 0.5  # Default for other types
        else:
            value_cost = 0.0  # No value cost if types are different
        
        # Combine costs
        cost = type_cost + value_cost * 0.5  # Weight value cost less than type cost
        
        # Apply path-based weighting - consider both paths
        # Use the average of both path weights for symmetry
        path_weight1 = self.calculate_path_weight(node1.path)
        path_weight2 = self.calculate_path_weight(node2.path)
        avg_path_weight = (path_weight1 + path_weight2) / 2.0
        
        cost *= avg_path_weight
        
        return cost
    
    def _longest_common_subsequence(self, s1: str, s2: str) -> int:
        """Calculate length of longest common subsequence between two strings."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
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
        
        if ZSS_AVAILABLE:
            # Use Zhang-Shasha algorithm from zss
            distance = zss.distance(
                tree1, tree2,
                get_children=lambda x: x.get_children(),
                get_label=lambda x: x.get_label(),
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
        else:
            # Use a simplified custom implementation
            distance, operations = self._custom_tree_edit_distance(tree1, tree2)
        
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
    
    def compare_nodes(self, node1: JsonNode, node2: JsonNode) -> Dict[str, Any]:
        """
        Compare two nodes and return detailed comparison information.
        Useful for debugging and understanding node equality decisions.
        
        Args:
            node1: First node
            node2: Second node
            
        Returns:
            Dictionary with comparison details
        """
        equal = self.are_nodes_equal(node1, node2)
        update_cost = self.update_cost(node1, node2)
        
        # Extract key names for clearer comparison
        key1 = node1.label.split('.')[-1] if '.' in node1.label else node1.label
        key2 = node2.label.split('.')[-1] if '.' in node2.label else node2.label
        
        return {
            "equal": equal,
            "update_cost": update_cost,
            "node1": {
                "path": node1.path,
                "key": key1,
                "type": node1.node_type,
                "value": node1.value,
                "has_children": len(node1.children) > 0
            },
            "node2": {
                "path": node2.path,
                "key": key2,
                "type": node2.node_type,
                "value": node2.value,
                "has_children": len(node2.children) > 0
            },
            "comparison": {
                "same_type": node1.node_type == node2.node_type,
                "same_key": key1 == key2,
                "same_value": node1.value == node2.value,
                "both_leaf": not node1.children and not node2.children,
                "both_array_indices": key1.startswith('[') and key2.startswith('[')
            }
        }
    
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
                formatted_ops.append({
                    'operation': 'update',
                    'path': node1.path,
                    'from_type': node1.node_type,
                    'to_type': node2.node_type,
                    'from_value': node1.value,
                    'to_value': node2.value
                })
        
        return formatted_ops
    
    def _custom_tree_edit_distance(self, tree1: JsonNode, tree2: JsonNode) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Simple custom tree edit distance implementation.
        This is a fallback if zss is not available.
        """
        # This is a simplified implementation that doesn't calculate the optimal edit script
        # It just compares the trees recursively and accumulates differences
        
        def compare_trees(node1, node2, path=""):
            # Base cost for this node comparison
            if node1.node_type != node2.node_type:
                # Type mismatch
                cost = self.type_change_cost.get((node1.node_type, node2.node_type), 1.0)
                operations.append({
                    'operation': 'update',
                    'path': path,
                    'from_type': node1.node_type,
                    'to_type': node2.node_type,
                    'from_value': node1.value,
                    'to_value': node2.value
                })
            elif node1.value != node2.value and node1.value is not None and node2.value is not None:
                # Value mismatch for leaf nodes
                cost = 0.5
                operations.append({
                    'operation': 'update',
                    'path': path,
                    'from_type': node1.node_type,
                    'to_type': node2.node_type,
                    'from_value': node1.value,
                    'to_value': node2.value
                })
            else:
                cost = 0.0
            
            # If both are leaf nodes, we're done
            if not node1.children and not node2.children:
                return cost
            
            # Get children by label
            children1 = {child.label.split('.')[-1]: child for child in node1.children}
            children2 = {child.label.split('.')[-1]: child for child in node2.children}
            
            # Find common and unique children
            common_labels = set(children1.keys()) & set(children2.keys())
            only_in_1 = set(children1.keys()) - common_labels
            only_in_2 = set(children2.keys()) - common_labels
            
            # Cost for missing/extra children
            for label in only_in_1:
                child = children1[label]
                child_cost = self.delete_cost(child)
                cost += child_cost
                operations.append({
                    'operation': 'remove',
                    'path': child.path,
                    'node_type': child.node_type,
                    'value': child.value
                })
            
            for label in only_in_2:
                child = children2[label]
                child_cost = self.insert_cost(child)
                cost += child_cost
                operations.append({
                    'operation': 'insert',
                    'path': child.path,
                    'node_type': child.node_type,
                    'value': child.value
                })
            
            # Recursive comparison for common children
            for label in common_labels:
                child1 = children1[label]
                child2 = children2[label]
                child_path = f"{path}.{label}" if path else label
                cost += compare_trees(child1, child2, child_path)
            
            return cost
        
        operations = []
        distance = compare_trees(tree1, tree2)
        return distance, operations
    
    def evaluate_structural_consistency(self, json_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate structural consistency across multiple JSON outputs.
        
        Args:
            json_outputs: List of JSON objects to evaluate
            
        Returns:
            Dictionary with consistency metrics
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
                    similarities.append((i, j, 0.0))
                    operations_by_pair[(i, j)] = []
        
        # Calculate average similarity
        avg_similarity = sum(sim for _, _, sim in similarities) / len(similarities) if similarities else 1.0
        
        # Calculate standard deviation
        std_similarity = np.std([sim for _, _, sim in similarities]) if len(similarities) > 1 else 0.0
        
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
        
        # Calculate consistency score (1 = perfect consistency)
        consistency_score = avg_similarity
        
        # Prepare detailed report
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "num_outputs_analyzed": n,
            "structural_consistency_score": consistency_score,
            "std_deviation": float(std_similarity),
            "min_similarity": min(sim for _, _, sim in similarities) if similarities else 1.0,
            "max_similarity": max(sim for _, _, sim in similarities) if similarities else 1.0,
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
            ],
            "perfect_consistency": consistency_score > 0.99
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
    for output in outputs:
        if isinstance(output, str):
            try:
                parsed.append(json.loads(output))
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON string: {output[:100]}...")
        elif isinstance(output, dict):
            parsed.append(output)
        else:
            print(f"Warning: Skipping non-JSON output of type {type(output)}")
    
    return parsed


def evaluate_json_structural_consistency(
    outputs: List[Union[str, Dict]],
    array_order_matters: bool = True,
    required_fields: List[str] = None
) -> Dict[str, Any]:
    """
    Evaluate structural consistency of JSON outputs.
    
    Args:
        outputs: List of JSON strings or dictionaries
        array_order_matters: Whether array order matters for comparison
        required_fields: List of required field paths
        
    Returns:
        Dictionary with consistency metrics
    """
    # Parse outputs
    parsed_outputs = parse_json_outputs(outputs)
    
    # Create evaluator
    evaluator = JsonTreeConsistencyEvaluator(
        array_order_matters=array_order_matters,
        required_fields=set(required_fields) if required_fields else set()
    )
    
    # Evaluate consistency
    return evaluator.evaluate_structural_consistency(parsed_outputs)


if __name__ == "__main__":
    # Example usage
    json1 = {
        "name": "John",
        "age": 30,
        "address": {
            "street": "123 Main St",
            "city": "New York"
        },
        "hobbies": ["reading", "swimming"]
    }
    
    json2 = {
        "name": "John",
        "age": 30,
        "address": {
            "street": "123 Main St",
            "city": "New York",
            "zip": "10001"  # Extra field
        },
        "hobbies": ["swimming", "reading"]  # Reordered
    }
    
    json3 = {
        "name": "John",
        "age": "30",  # String instead of number
        "address": {
            "street": "123 Main Street",  # Slightly different
            "city": "New York"
        }
        # Missing hobbies
    }
    
    # Evaluate consistency
    result = evaluate_json_structural_consistency(
        [json1, json2, json3],
        array_order_matters=False,
        required_fields=["name", "age", "address.city"]
    )
    
    # Print results
    print(f"Structural Consistency Score: {result['structural_consistency_score']:.4f}")
    print(f"Perfect Consistency: {result['perfect_consistency']}")
    print("\nMost Different Pairs:")
    for pair_info in result["most_different_pairs"]:
        print(f"  Pair {pair_info['pair']}: Similarity {pair_info['similarity']:.4f}")
    
    print("\nFrequently Edited Paths:")
    for path_info in result["frequently_edited_paths"][:5]:
        print(f"  {path_info['path']}: {path_info['edit_count']} edits")