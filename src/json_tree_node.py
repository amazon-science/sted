"""
JSON tree node representation for structural analysis.
"""

from typing import Any, List


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
    
    def count_nodes(self) -> int:
        """Count the total number of nodes in the tree."""
        count = 1 if self.value else 0 # Count the current node
        for child in self.get_children():
            count += child.count_nodes()
        
        if self.node_type == "array" and len(self.value) > 0 and isinstance(self.value[0],dict):
            for el in self.value:
                el_tree = JsonNode.from_dict(el)
                count += el_tree.count_nodes()
        return count
    
    @classmethod
    def from_dict(cls, data: dict, path: str = "", sort_keys: bool = True, sort_arrays: bool = True) -> 'JsonNode':
        """
        Create a JsonNode tree from a dictionary.

        Args:
            data: The dictionary to convert
            path: The current path in the JSON structure
            sort_keys: Whether to sort dictionary keys for deterministic behavior
            sort_arrays: Whether to sort array elements for order invariance

        Returns:
            A JsonNode representing the root of the tree
        """    
        full_path = path if path else "root"
        if isinstance(data, dict):
            # Create a node for the object
            node = cls(full_path, node_type="object")
            
            # Add children for each key-value pair
            items = sorted(data.items()) if sort_keys else data.items()
            for key, value in items:
                child_path = f"{full_path}.{key}" if full_path != "root" else key
                child = cls.from_dict(value, child_path, sort_keys, sort_arrays)
                node.add_child(child)
        elif isinstance(data, list):
            # Sort array elements for order invariance if requested
            if sort_arrays:
                try:
                    sorted_data = sorted(data, key=lambda x: str(x))
                except TypeError:
                    sorted_data = data
            else:
                sorted_data = data
            # Create a leaf node with array as value
            node = cls(full_path, sorted_data)
        else:
            # Create a leaf node for primitive values
            node = cls(full_path, data)
        
        return node
    
    def reconstruct_json(self):
        """Reconstruct JSON from tree node"""
        if self.node_type == "object":
            result = {}
            for child in self.children:
                key = child.label.split('.')[-1]
                result[key] = child.reconstruct_json()
            return result
        else:
            return self.value