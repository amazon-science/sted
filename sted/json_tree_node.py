"""
JSON tree node representation for structural analysis.
"""

from typing import Any


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
        count = 1  # Count the current node
        for child in self.get_children():
            count += child.count_nodes()
        return count

    @classmethod
    def from_dict(cls, data: dict, path: str = "", sort_keys: bool = True, sort_arrays: bool = True,
                  order_sensitive_fields: set = None) -> 'JsonNode':
        """
        Create a JsonNode tree from a dictionary.

        Args:
            data: The dictionary to convert
            path: The current path in the JSON structure
            sort_keys: Whether to sort dictionary keys for deterministic behavior
            sort_arrays: Whether to sort array elements for order invariance
            order_sensitive_fields: Set of field names where array order matters (not sorted)

        Returns:
            A JsonNode representing the root of the tree
        """
        order_sensitive_fields = order_sensitive_fields or set()
        full_path = path if path else "root"

        if isinstance(data, dict):
            # Create a node for the object
            node = cls(full_path, node_type="object")

            # Add children for each key-value pair
            items = sorted(data.items()) if sort_keys else data.items()
            for key, value in items:
                child_path = f"{full_path}.{key}" if full_path != "root" else key
                child = cls.from_dict(value, child_path, sort_keys, sort_arrays, order_sensitive_fields)
                node.add_child(child)
        elif isinstance(data, list):
            # Extract the field name from the path (e.g., "calls" from "root.calls" or "calls[0]")
            field_name = full_path.split('.')[-1].split('[')[0]

            # Check if this field is order-sensitive
            is_order_sensitive = field_name in order_sensitive_fields

            # Sort array elements for order invariance if requested AND not order-sensitive
            if sort_arrays and not is_order_sensitive:
                try:
                    sorted_data = sorted(data, key=lambda x: str(x))
                except TypeError:
                    sorted_data = data
            else:
                sorted_data = data

            # Create an array node with children for each element
            node = cls(full_path, node_type="array")
            node.value = sorted_data  # Keep value for backward compatibility

            # Create child nodes for each array element
            for idx, item in enumerate(sorted_data):
                child_path = f"{full_path}[{idx}]"
                if isinstance(item, (dict, list)):
                    # Recursively create tree for complex elements
                    child = cls.from_dict(item, child_path, sort_keys, sort_arrays, order_sensitive_fields)
                else:
                    # Create leaf node for primitive elements
                    child = cls(child_path, item)
                node.add_child(child)
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
        elif self.node_type == "array":
            # Reconstruct array from children
            return [child.reconstruct_json() for child in self.children]
        else:
            return self.value
