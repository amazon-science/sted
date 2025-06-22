import unittest
import numpy as np
from unittest.mock import Mock, MagicMock
import json
import asyncio
import pandas as pd
from consistency_eval_fixed import FieldAwareConsistencyCalculator

class TestFieldAwareConsistencyCalculator(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock bedrock client
        self.mock_bedrock_client = Mock()
        self.mock_bedrock_client.invoke_model.return_value = {
            "body": MagicMock(read=lambda: json.dumps({"embedding": [0.1] * 256}).encode())
        }
        
        # Initialize calculator with nested field support
        self.calculator = FieldAwareConsistencyCalculator(
            bedrock_client=self.mock_bedrock_client,
            eval_fields=[
                "error_type",
                "user.name",
                "user.address.city",
                "metadata.severity",
                "tags.0"  # First tag in a list
            ],
            result_field_name="data.corrections",
            primary_field="error_type",
            nested_field_separator="."
        )
        
    def test_get_nested_value_simple(self):
        """Test getting simple non-nested values"""
        data = {"name": "John", "age": 30}
        
        self.assertEqual(self.calculator._get_nested_value(data, "name"), "John")
        self.assertEqual(self.calculator._get_nested_value(data, "age"), 30)
        self.assertIsNone(self.calculator._get_nested_value(data, "missing"))
        
    def test_get_nested_value_nested(self):
        """Test getting nested values"""
        data = {
            "user": {
                "name": "John",
                "address": {
                    "city": "Boston",
                    "zip": "02101"
                }
            }
        }
        
        self.assertEqual(self.calculator._get_nested_value(data, "user.name"), "John")
        self.assertEqual(self.calculator._get_nested_value(data, "user.address.city"), "Boston")
        self.assertEqual(self.calculator._get_nested_value(data, "user.address.zip"), "02101")
        self.assertIsNone(self.calculator._get_nested_value(data, "user.address.country"))
        
    def test_get_nested_value_with_list(self):
        """Test getting values from lists"""
        data = {
            "items": ["first", "second", "third"],
            "users": [
                {"name": "John", "age": 30},
                {"name": "Jane", "age": 25}
            ]
        }
        
        self.assertEqual(self.calculator._get_nested_value(data, "items.0"), "first")
        self.assertEqual(self.calculator._get_nested_value(data, "items.2"), "third")
        self.assertEqual(self.calculator._get_nested_value(data, "users.0.name"), "John")
        self.assertEqual(self.calculator._get_nested_value(data, "users.1.age"), 25)
        self.assertIsNone(self.calculator._get_nested_value(data, "items.10"))
        
    def test_extract_fields_flat(self):
        """Test field extraction from flat structure"""
        # Create calculator for flat structure
        calc_flat = FieldAwareConsistencyCalculator(
            bedrock_client=self.mock_bedrock_client,
            eval_fields=["error_type", "severity", "message"],
            result_field_name="corrections",
            primary_field="error_type"
        )
        
        response = {
            "corrections": [
                {"error_type": "typo", "severity": "low", "message": "Spelling error"},
                {"error_type": "grammar", "severity": "medium", "message": "Grammar issue"}
            ]
        }
        
        fields = calc_flat.extract_fields(response)
        
        self.assertEqual(len(fields["error_type"]), 2)
        self.assertEqual(fields["error_type"], ["typo", "grammar"])
        self.assertEqual(fields["severity"], ["low", "medium"])
        
    def test_extract_fields_nested(self):
        """Test field extraction from nested structure"""
        response = {
            "data": {
                "corrections": [
                    {
                        "error_type": "typo",
                        "user": {
                            "name": "John Doe",
                            "address": {"city": "Boston"}
                        },
                        "metadata": {"severity": "low"},
                        "tags": ["spelling", "minor"]
                    },
                    {
                        "error_type": "grammar",
                        "user": {
                            "name": "Jane Smith",
                            "address": {"city": "New York"}
                        },
                        "metadata": {"severity": "high"},
                        "tags": ["syntax", "major"]
                    }
                ]
            }
        }
        
        fields = self.calculator.extract_fields(response)
        
        self.assertEqual(fields["error_type"], ["typo", "grammar"])
        self.assertEqual(fields["user.name"], ["John Doe", "Jane Smith"])
        self.assertEqual(fields["user.address.city"], ["Boston", "New York"])
        self.assertEqual(fields["metadata.severity"], ["low", "high"])
        self.assertEqual(fields["tags.0"], ["spelling", "syntax"])
        
    def test_extract_fields_missing_data(self):
        """Test field extraction with missing nested data"""
        response = {
            "data": {
                "corrections": [
                    {
                        "error_type": "typo",
                        "user": {"name": "John"},  # Missing address
                        "metadata": {"severity": "low"}
                        # Missing tags
                    },
                    {
                        "error_type": "grammar",
                        # Missing user entirely
                        "metadata": {"severity": "high"},
                        "tags": ["syntax"]
                    }
                ]
            }
        }
        
        fields = self.calculator.extract_fields(response)
        
        self.assertEqual(fields["error_type"], ["typo", "grammar"])
        self.assertEqual(fields["user.name"], ["John"])  # Only one name
        self.assertEqual(fields["user.address.city"], [])  # No cities
        self.assertEqual(fields["metadata.severity"], ["low", "high"])
        self.assertEqual(fields["tags.0"], ["syntax"])  # Only one first tag
        
    def test_calculate_field_consistency_nested(self):
        """Test consistency calculation with nested fields"""
        responses = [
            {
                "data": {
                    "corrections": [
                        {
                            "error_type": "typo",
                            "user": {"name": "John Doe", "address": {"city": "Boston"}},
                            "metadata": {"severity": "low"}
                        }
                    ]
                }
            },
            {
                "data": {
                    "corrections": [
                        {
                            "error_type": "typo",
                            "user": {"name": "John Doe", "address": {"city": "Boston"}},
                            "metadata": {"severity": "low"}
                        }
                    ]
                }
            },
            {
                "data": {
                    "corrections": [
                        {
                            "error_type": "grammar",
                            "user": {"name": "Jane Smith", "address": {"city": "New York"}},
                            "metadata": {"severity": "high"}
                        }
                    ]
                }
            }
        ]
        
        # Mock embedding responses for consistency
        self.calculator.embedding_cache = {
            "typo": [0.1] * 256,
            "grammar": [0.9] * 256,
            "John Doe": [0.2] * 256,
            "Jane Smith": [0.8] * 256,
            "Boston": [0.3] * 256,
            "New York": [0.7] * 256,
            "low": [0.4] * 256,
            "high": [0.6] * 256
        }
        
        score, metrics = self.calculator.calculate_field_consistency(responses, "error_type")
        
        # Should have some consistency but not perfect (2 typos, 1 grammar)
        self.assertGreater(score, 0)
        self.assertLess(score, 1)
        
        # Test perfect consistency - FIX: Use assertAlmostEqual for floating point comparison
        score_city, _ = self.calculator.calculate_field_consistency(responses[:2], "user.address.city")
        self.assertAlmostEqual(score_city, 1.0, places=10)  # Both have Boston
        
    def test_analyze_corrections_consistency_nested(self):
        """Test analyzing corrections consistency with nested data"""
        data_list = [
            {
                "data": {
                    "corrections": [
                        {
                            "error_type": "typo",
                            "user": {"name": "John Doe"},
                            "fix": "Fixed spelling"
                        }
                    ]
                }
            },
            {
                "data": {
                    "corrections": [
                        {
                            "error_type": "typo",
                            "user": {"name": "John D."},
                            "fix": "Corrected spelling"
                        }
                    ]
                }
            },
            {
                "data": {
                    "corrections": [
                        {
                            "error_type": "grammar",
                            "user": {"name": "Jane Smith"},
                            "fix": "Fixed grammar"
                        }
                    ]
                }
            }
        ]
        
        # Mock similar embeddings for "typo" variations
        self.calculator.embedding_cache = {
            "typo": [0.1] * 256,
            "grammar": [0.9] * 256,
        }
        
        result = self.calculator.analyze_corrections_consistency(data_list)
        
        self.assertIn("total_corrections", result)
        self.assertIn("common_primary_values", result)
        self.assertIn("analysis", result)
        
    def test_calculate_all_metrics_nested(self):
        """Test full metrics calculation with nested data"""
        # Prediction data with nested structure
        pred_dict = {
            "doc1": [
                {
                    "data": {
                        "corrections": [
                            {"error_type": "typo", "location": {"line": 10}},
                            {"error_type": "grammar", "location": {"line": 20}}
                        ]
                    }
                },
                {
                    "data": {
                        "corrections": [
                            {"error_type": "typo", "location": {"line": 10}},
                            {"error_type": "grammar", "location": {"line": 21}}  # Slightly different
                        ]
                    }
                }
            ],
            "doc2": [
                {
                    "data": {
                        "corrections": [
                            {"error_type": "syntax", "location": {"line": 5}}
                        ]
                    }
                }
            ]
        }
        
        # Ground truth dataframe
        gt_df = pd.DataFrame([
            {"document_id": "doc1", "error_type": "typo"},
            {"document_id": "doc1", "error_type": "grammar"},
            {"document_id": "doc2", "error_type": "syntax"},
            {"document_id": "doc2", "error_type": "formatting"}  # This one is missing in predictions
        ])
        
        # Mock embeddings
        self.calculator.embedding_cache = {
            "typo": [0.1] * 256,
            "grammar": [0.2] * 256,
            "syntax": [0.3] * 256,
            "formatting": [0.4] * 256
        }
        
        # Use only fields that are actually in the consistency scores
        eval_fields = ["error_type"]
        all_metrics, overall_metrics, pred_results = self.calculator.calculate_all_metrics(
            pred_dict, gt_df, eval_fields
        )
        
        # Check that metrics were calculated
        self.assertIn("doc1", all_metrics)
        self.assertIn("doc2", all_metrics)
        self.assertIn("precision", overall_metrics)
        self.assertIn("recall", overall_metrics)
        self.assertIn("f1", overall_metrics)
        
    def test_set_nested_value(self):
        """Test setting nested values"""
        data = {}
        
        self.calculator._set_nested_value(data, "simple", "value")
        self.assertEqual(data["simple"], "value")
        
        self.calculator._set_nested_value(data, "user.name", "John")
        self.assertEqual(data["user"]["name"], "John")
        
        self.calculator._set_nested_value(data, "user.address.city", "Boston")
        self.assertEqual(data["user"]["address"]["city"], "Boston")
        
        # Test overwriting
        self.calculator._set_nested_value(data, "user.name", "Jane")
        self.assertEqual(data["user"]["name"], "Jane")
        
    def test_empty_and_none_handling(self):
        """Test handling of empty and None values"""
        # Test with None response
        fields = self.calculator.extract_fields(None)
        for field in self.calculator.fields:
            self.assertEqual(fields[field], [])
        
        # Test with empty corrections
        response = {"data": {"corrections": []}}
        fields = self.calculator.extract_fields(response)
        for field in self.calculator.fields:
            self.assertEqual(fields[field], [])
        
        # Test with missing nested path
        response = {"wrong": {"path": []}}
        fields = self.calculator.extract_fields(response)
        for field in self.calculator.fields:
            self.assertEqual(fields[field], [])

    @unittest.skipIf(not hasattr(asyncio, 'run'), "Async tests require Python 3.7+")
    def test_async_methods(self):
        """Test async versions of methods"""
        async def run_async_test():
            responses = [
                {
                    "data": {
                        "corrections": [
                            {"error_type": "typo", "user": {"name": "John"}}
                        ]
                    }
                }
            ]
            
            # Mock async embedding
            self.calculator.embedding_cache = {
                "typo": [0.1] * 256,
                "John": [0.2] * 256
            }
            
            results, metrics = await self.calculator.calculate_prompt_consistency_async(responses)
            
            self.assertIn("consistency_score", results)
            self.assertIn("overall", results["consistency_score"])
            
        asyncio.run(run_async_test())

# Run the tests
if __name__ == "__main__":
    unittest.main()
