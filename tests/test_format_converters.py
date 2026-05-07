"""Tests for format converters (XML, YAML, HTML -> dict) and multi-format parsing."""

import pytest
from sted.format_converters import (
    detect_format,
    xml_to_dict,
    yaml_to_dict,
    html_to_dict,
    parse_structured_string,
)
from sted.utils import parse_structured_outputs


# --- detect_format ---

class TestDetectFormat:
    def test_json_object(self):
        assert detect_format('{"key": "value"}') == "json"

    def test_json_array(self):
        assert detect_format('[1, 2, 3]') == "json"

    def test_json_with_whitespace(self):
        assert detect_format('  \n  {"key": "value"}  ') == "json"

    def test_xml_with_declaration(self):
        assert detect_format('<?xml version="1.0"?><root/>') == "xml"

    def test_xml_element(self):
        assert detect_format("<root><child>text</child></root>") == "xml"

    def test_yaml_with_separator(self):
        assert detect_format("---\nkey: value\n") == "yaml"

    def test_yaml_key_value(self):
        assert detect_format("name: John\nage: 30\n") == "yaml"

    def test_html(self):
        assert detect_format("<!DOCTYPE html><html><body></body></html>") == "html"

    def test_unknown(self):
        assert detect_format("just some random text") == "unknown"


# --- xml_to_dict ---

class TestXmlToDict:
    def test_simple_element(self):
        xml = "<user><name>John</name><age>30</age></user>"
        result = xml_to_dict(xml)
        assert result == {"user": {"name": "John", "age": "30"}}

    def test_attributes(self):
        xml = '<user id="1"><name>John</name></user>'
        result = xml_to_dict(xml)
        assert result == {"user": {"@id": "1", "name": "John"}}

    def test_repeated_children_become_list(self):
        xml = "<users><user>Alice</user><user>Bob</user></users>"
        result = xml_to_dict(xml)
        assert result == {"users": {"user": ["Alice", "Bob"]}}

    def test_nested_structure(self):
        xml = "<root><parent><child>value</child></parent></root>"
        result = xml_to_dict(xml)
        assert result == {"root": {"parent": {"child": "value"}}}

    def test_empty_element(self):
        xml = "<root><empty/></root>"
        result = xml_to_dict(xml)
        assert result == {"root": {"empty": ""}}

    def test_mixed_content(self):
        xml = '<item type="book"><title>Python</title></item>'
        result = xml_to_dict(xml)
        assert result == {"item": {"@type": "book", "title": "Python"}}

    def test_namespace_stripped(self):
        xml = '<root xmlns:ns="http://example.com"><ns:child>val</ns:child></root>'
        result = xml_to_dict(xml)
        assert "child" in result["root"]

    def test_invalid_xml_raises(self):
        with pytest.raises(ValueError, match="Invalid XML"):
            xml_to_dict("<unclosed>")

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            xml_to_dict(123)


# --- yaml_to_dict ---

class TestYamlToDict:
    def test_simple_mapping(self):
        yaml_str = "name: John\nage: 30\n"
        result = yaml_to_dict(yaml_str)
        assert result == {"name": "John", "age": 30}

    def test_nested_mapping(self):
        yaml_str = "user:\n  name: John\n  address:\n    city: NYC\n"
        result = yaml_to_dict(yaml_str)
        assert result == {"user": {"name": "John", "address": {"city": "NYC"}}}

    def test_list_values(self):
        yaml_str = "colors:\n  - red\n  - green\n  - blue\n"
        result = yaml_to_dict(yaml_str)
        assert result == {"colors": ["red", "green", "blue"]}

    def test_empty_yaml(self):
        result = yaml_to_dict("")
        assert result == {}

    def test_scalar_wrapped(self):
        result = yaml_to_dict("42")
        assert result == {"root": 42}

    def test_with_document_separator(self):
        yaml_str = "---\nkey: value\n"
        result = yaml_to_dict(yaml_str)
        assert result == {"key": "value"}

    def test_invalid_yaml_raises(self):
        with pytest.raises(ValueError, match="Invalid YAML"):
            yaml_to_dict(":\n  :\n    - ][invalid")


# --- html_to_dict ---

class TestHtmlToDict:
    def test_simple_html(self):
        html = "<div><p>Hello</p></div>"
        result = html_to_dict(html)
        assert "div" in result
        assert result["div"]["p"] == "Hello"

    def test_html_attributes(self):
        html = '<a href="https://example.com">Link</a>'
        result = html_to_dict(html)
        assert result["a"]["@href"] == "https://example.com"
        assert result["a"]["#text"] == "Link"

    def test_repeated_tags(self):
        html = "<ul><li>A</li><li>B</li></ul>"
        result = html_to_dict(html)
        assert result["ul"]["li"] == ["A", "B"]


# --- parse_structured_string ---

class TestParseStructuredString:
    def test_json_explicit(self):
        result = parse_structured_string('{"key": "value"}', format="json")
        assert result == {"key": "value"}

    def test_xml_explicit(self):
        result = parse_structured_string("<root><k>v</k></root>", format="xml")
        assert result == {"root": {"k": "v"}}

    def test_yaml_explicit(self):
        result = parse_structured_string("key: value", format="yaml")
        assert result == {"key": "value"}

    def test_auto_detect_json(self):
        result = parse_structured_string('{"auto": true}')
        assert result == {"auto": True}

    def test_auto_detect_xml(self):
        result = parse_structured_string("<auto>true</auto>")
        assert result == {"auto": "true"}

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Could not detect format"):
            parse_structured_string("random text with no structure")

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported format"):
            parse_structured_string("{}", format="toml")

    def test_json_array_wrapped(self):
        result = parse_structured_string("[1, 2, 3]", format="json")
        assert result == {"root": [1, 2, 3]}


# --- parse_structured_outputs ---

class TestParseStructuredOutputs:
    def test_json_delegates_to_parse_json_outputs(self):
        outputs = ['{"a": 1}', {"b": 2}]
        result = parse_structured_outputs(outputs, format="json")
        assert result == [{"a": 1}, {"b": 2}]

    def test_xml_format(self):
        outputs = ["<root><key>val</key></root>"]
        result = parse_structured_outputs(outputs, format="xml")
        assert result == [{"root": {"key": "val"}}]

    def test_yaml_format(self):
        outputs = ["key: value"]
        result = parse_structured_outputs(outputs, format="yaml")
        assert result == [{"key": "value"}]

    def test_dict_passthrough(self):
        outputs = [{"already": "parsed"}]
        result = parse_structured_outputs(outputs, format="xml")
        assert result == [{"already": "parsed"}]

    def test_invalid_string_warns(self):
        with pytest.warns(match="Could not parse"):
            result = parse_structured_outputs(["<unclosed>"], format="xml")
        assert result == []


# --- Cross-format equivalence (key for NeurIPS paper) ---

class TestCrossFormatEquivalence:
    """Verify that the same data produces equivalent dicts across formats."""

    def test_json_xml_yaml_produce_same_structure(self):
        json_str = '{"name": "Alice", "age": "30"}'
        xml_str = "<root><name>Alice</name><age>30</age></root>"
        yaml_str = "name: Alice\nage: '30'\n"

        json_dict = parse_structured_string(json_str, format="json")
        xml_dict = parse_structured_string(xml_str, format="xml")["root"]
        yaml_dict = parse_structured_string(yaml_str, format="yaml")

        # All should have same keys and values
        assert json_dict["name"] == xml_dict["name"] == yaml_dict["name"] == "Alice"
        assert json_dict["age"] == xml_dict["age"] == yaml_dict["age"] == "30"

    def test_nested_equivalence(self):
        json_str = '{"user": {"name": "Bob", "city": "NYC"}}'
        xml_str = "<root><user><name>Bob</name><city>NYC</city></user></root>"
        yaml_str = "user:\n  name: Bob\n  city: NYC\n"

        json_dict = parse_structured_string(json_str, format="json")
        xml_dict = parse_structured_string(xml_str, format="xml")["root"]
        yaml_dict = parse_structured_string(yaml_str, format="yaml")

        assert json_dict["user"]["name"] == xml_dict["user"]["name"] == "Bob"
        assert json_dict["user"]["city"] == xml_dict["user"]["city"] == "NYC"
