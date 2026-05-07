"""
Format converters for structured data formats (XML, YAML, HTML).

Converts various structured formats to Python dict/list representations
that can be consumed by STED's format-agnostic tree comparison engine.
"""

import json
import re
import warnings
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Union


def detect_format(input_string: str) -> str:
    """
    Detect the format of a structured data string.

    Args:
        input_string: Raw string to detect format of

    Returns:
        One of "json", "xml", "yaml", "html", or "unknown"
    """
    stripped = input_string.strip()

    # JSON: starts with { or [
    if stripped.startswith(("{", "[")):
        try:
            json.loads(stripped)
            return "json"
        except (json.JSONDecodeError, ValueError):
            pass

    # XML: starts with < and contains closing tags
    if stripped.startswith("<") and not stripped.startswith("<!DOCTYPE"):
        # Check for XML declaration or root element
        if stripped.startswith("<?xml") or re.match(r"<[a-zA-Z]", stripped):
            return "xml"

    # HTML: starts with <!DOCTYPE or <html
    if stripped.lower().startswith(("<!doctype", "<html")):
        return "html"

    # YAML: check for common YAML patterns
    # YAML typically has key: value on lines, or starts with ---
    if stripped.startswith("---") or re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*\s*:", stripped):
        return "yaml"

    return "unknown"


def xml_to_dict(xml_input: Union[str, ET.Element]) -> Dict[str, Any]:
    """
    Convert an XML string or ElementTree Element to a nested dictionary.

    Handles attributes, text content, and nested elements. Follows conventions:
    - Element attributes are stored with '@' prefix
    - Text content is stored as '#text' if element also has children/attributes
    - Multiple children with same tag become a list

    Args:
        xml_input: XML string or ElementTree Element

    Returns:
        Dictionary representation of the XML structure
    """
    if isinstance(xml_input, str):
        xml_input = xml_input.strip()
        try:
            root = ET.fromstring(xml_input)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML: {e}")
    elif isinstance(xml_input, ET.Element):
        root = xml_input
    else:
        raise TypeError(f"Expected str or Element, got {type(xml_input)}")

    return {_clean_tag(root.tag): _element_to_dict(root)}


def _clean_tag(tag: str) -> str:
    """Remove XML namespace from tag if present."""
    # {namespace}tagname -> tagname
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _element_to_dict(element: ET.Element) -> Any:
    """
    Recursively convert an XML Element to a dict.

    Args:
        element: XML Element to convert

    Returns:
        Dict, list, or string representation
    """
    result = {}

    # Add attributes with '@' prefix
    for attr_name, attr_value in element.attrib.items():
        result[f"@{_clean_tag(attr_name)}"] = attr_value

    # Group children by tag name
    children_by_tag: Dict[str, List] = {}
    for child in element:
        tag = _clean_tag(child.tag)
        if tag not in children_by_tag:
            children_by_tag[tag] = []
        children_by_tag[tag].append(child)

    # Add children
    for tag, children in children_by_tag.items():
        if len(children) == 1:
            result[tag] = _element_to_dict(children[0])
        else:
            result[tag] = [_element_to_dict(child) for child in children]

    # Handle text content
    text = (element.text or "").strip()
    if text:
        if result:
            # Element has both children/attributes and text
            result["#text"] = text
        else:
            # Element only has text content - return as string
            return text

    # If no children, attributes, or text, return empty string
    if not result:
        return ""

    return result


def yaml_to_dict(yaml_string: str) -> Dict[str, Any]:
    """
    Convert a YAML string to a nested dictionary.

    Requires PyYAML to be installed.

    Args:
        yaml_string: YAML-formatted string

    Returns:
        Dictionary representation of the YAML structure

    Raises:
        ImportError: If PyYAML is not installed
        ValueError: If YAML is invalid
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for YAML support. Install with: pip install pyyaml"
        )

    try:
        result = yaml.safe_load(yaml_string)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML: {e}")

    if result is None:
        return {}
    if not isinstance(result, dict):
        return {"root": result}
    return result


def html_to_dict(html_string: str) -> Dict[str, Any]:
    """
    Convert an HTML string to a nested dictionary.

    Uses Python's html.parser for lightweight parsing without external dependencies.
    For production use with complex HTML, consider BeautifulSoup.

    Args:
        html_string: HTML-formatted string

    Returns:
        Dictionary representation of the HTML structure
    """
    from html.parser import HTMLParser

    class _HTMLToDict(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack = [{"tag": "root", "children": [], "attrs": {}}]

        def handle_starttag(self, tag, attrs):
            node = {"tag": tag, "children": [], "attrs": dict(attrs)}
            self.stack[-1]["children"].append(node)
            self.stack.append(node)

        def handle_endtag(self, tag):
            if len(self.stack) > 1:
                self.stack.pop()

        def handle_data(self, data):
            text = data.strip()
            if text:
                self.stack[-1]["children"].append(text)

        def get_result(self) -> Dict[str, Any]:
            return _html_node_to_dict(self.stack[0])

    def _html_node_to_dict(node: dict) -> Dict[str, Any]:
        if isinstance(node, str):
            return node
        result = {}
        if node.get("attrs"):
            for k, v in node["attrs"].items():
                result[f"@{k}"] = v
        children_by_tag: Dict[str, list] = {}
        texts = []
        for child in node.get("children", []):
            if isinstance(child, str):
                texts.append(child)
            else:
                tag = child["tag"]
                if tag not in children_by_tag:
                    children_by_tag[tag] = []
                children_by_tag[tag].append(_html_node_to_dict(child))
        for tag, items in children_by_tag.items():
            result[tag] = items[0] if len(items) == 1 else items
        if texts:
            text = " ".join(texts)
            if result:
                result["#text"] = text
            elif not result:
                return text
        return result if result else ""

    parser = _HTMLToDict()
    parser.feed(html_string)
    return parser.get_result()


def parse_structured_string(
    input_string: str, format: Optional[str] = None
) -> Dict[str, Any]:
    """
    Parse a structured data string into a Python dictionary.

    Auto-detects format if not specified.

    Args:
        input_string: The structured data string to parse
        format: One of "json", "xml", "yaml", "html", or None for auto-detect

    Returns:
        Parsed dictionary representation

    Raises:
        ValueError: If format cannot be detected or parsing fails
    """
    if format is None:
        format = detect_format(input_string)
        if format == "unknown":
            raise ValueError(
                "Could not detect format. Please specify format='json'|'xml'|'yaml'|'html'"
            )

    format = format.lower()

    if format == "json":
        try:
            result = json.loads(input_string)
            if not isinstance(result, dict):
                result = {"root": result}
            return result
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
    elif format == "xml":
        return xml_to_dict(input_string)
    elif format in ("yaml", "yml"):
        return yaml_to_dict(input_string)
    elif format == "html":
        return html_to_dict(input_string)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json', 'xml', 'yaml', or 'html'")
