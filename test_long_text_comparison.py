"""
Test script for enhanced long text comparison in semantic JSON tree consistency

This script demonstrates how the enhanced string comparison handles long text values
by breaking them into chunks and finding optimal matches between them.
"""

import json
import time
from semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator

def print_separator():
    print("\n" + "="*60 + "\n")

def test_long_text_comparison():
    """Test the enhanced string comparison with long text values."""
    print("Testing long text comparison...")
    
    # Create evaluator with different settings
    evaluator_standard = SemanticJsonTreeConsistencyEvaluator(
        use_semantic_similarity=False,
        string_method='levenshtein'
    )
    
    evaluator_semantic = SemanticJsonTreeConsistencyEvaluator(
        use_semantic_similarity=True,
        string_method='semantic'
    )
    
    # Test case 1: Paragraphs with similar content but different wording
    text1 = """
    The JSON tree consistency evaluator is designed to measure structural similarity
    between JSON objects. It converts JSON structures to trees and calculates edit
    distances between them. This approach allows for accurate comparison of complex
    nested structures, taking into account both the structure and the values.
    
    The evaluator supports various configuration options, including schema awareness,
    array order sensitivity, and path weight decay. It can also be customized with
    different cost functions for type changes and required fields.
    """
    
    text2 = """
    Our JSON structural similarity tool measures how alike two JSON objects are by
    transforming them into tree structures and computing the edit distance. This method
    enables precise comparison of nested JSON data, considering both structural elements
    and actual values within the objects.
    
    The tool is highly configurable with options for schema validation, handling array
    ordering, and adjusting weights based on path depth. Users can also define custom
    cost functions for different types of changes and specify which fields are required.
    """
    
    # Test case 2: Code blocks with similar functionality but different variable names
    code1 = """
    function calculateDistance(json1, json2) {
        // Convert JSONs to trees
        const tree1 = jsonToTree(json1);
        const tree2 = jsonToTree(json2);
        
        // Calculate edit distance
        const distance = treeEditDistance(tree1, tree2);
        
        // Normalize by tree size
        const size1 = countNodes(tree1);
        const size2 = countNodes(tree2);
        const maxSize = Math.max(size1, size2);
        
        return distance / maxSize;
    }
    """
    
    code2 = """
    function computeSimilarity(obj1, obj2) {
        // Transform objects to tree representation
        const t1 = convertToTree(obj1);
        const t2 = convertToTree(obj2);
        
        // Get the edit distance between trees
        const editDist = calculateTreeDistance(t1, t2);
        
        // Normalize by the larger tree size
        const nodeCount1 = getNodeCount(t1);
        const nodeCount2 = getNodeCount(t2);
        const largerSize = Math.max(nodeCount1, nodeCount2);
        
        return 1 - (editDist / largerSize);
    }
    """
    
    # Test case 3: Partially similar text with added/removed sections
    text3 = """
    The JSON tree consistency evaluator is designed to measure structural similarity
    between JSON objects. It converts JSON structures to trees and calculates edit
    distances between them.
    
    This approach allows for accurate comparison of complex nested structures,
    taking into account both the structure and the values.
    """
    
    text4 = """
    The JSON tree consistency evaluator is designed to measure structural similarity
    between JSON objects. It converts JSON structures to trees and calculates edit
    distances between them.
    
    The evaluator has been enhanced with semantic understanding capabilities to recognize
    when fields have different names but similar meanings. This makes it more robust
    when comparing JSON objects with different naming conventions.
    """
    
    # Test case 4: Technical documentation with reordered sections
    doc1 = """
    # JSON Tree Consistency Evaluator
    
    ## Overview
    The JSON Tree Consistency Evaluator is a tool for measuring the structural similarity between JSON objects.
    It uses tree edit distance algorithms to calculate how similar two JSON structures are.
    
    ## Features
    - Converts JSON to tree structures
    - Calculates edit distance between trees
    - Supports custom cost functions
    - Handles array order sensitivity
    - Provides detailed similarity reports
    
    ## Configuration Options
    - `schema_aware`: Whether to use schema information
    - `array_order_matters`: Whether array order affects similarity
    - `path_weight_decay`: Weight decay factor for deeper paths
    - `type_change_cost`: Custom costs for type changes
    - `required_fields`: Set of required field paths
    
    ## Usage Example
    ```python
    evaluator = JsonTreeConsistencyEvaluator()
    similarity = evaluator.compare(json1, json2)
    print(f"Similarity: {similarity:.2f}")
    ```
    """
    
    doc2 = """
    # JSON Structural Similarity Tool
    
    ## Features
    - Tree-based JSON comparison
    - Customizable edit distance calculation
    - Detailed similarity reporting
    - Array order handling options
    - Schema-aware comparison mode
    
    ## Configuration Options
    - `schema_aware`: Enable/disable schema validation
    - `array_order_matters`: Consider/ignore array element order
    - `path_weight_decay`: Adjust importance by path depth
    - `type_change_cost`: Customize type conversion costs
    - `required_fields`: Specify critical fields
    
    ## Overview
    This tool evaluates how similar two JSON documents are by converting them to tree structures
    and applying tree edit distance algorithms. It's particularly useful for comparing complex
    nested JSON structures where simple string comparison would be inadequate.
    
    ## Usage Example
    ```python
    tool = JsonStructuralSimilarityTool()
    result = tool.calculate_similarity(json1, json2)
    print(f"Similarity score: {result:.2f}")
    ```
    """
    
    # Test case 5: Error logs with similar patterns but different details
    log1 = """
    [2025-06-18 14:32:15] ERROR: Failed to parse JSON input
    File: /app/services/parser.js
    Line: 127
    Details: Unexpected token } in JSON at position 1024
    
    Stack trace:
    at JSON.parse (<anonymous>)
    at Parser.parseInput (/app/services/parser.js:127:23)
    at RequestHandler.processRequest (/app/handlers/request.js:45:18)
    at Server.handleConnection (/app/server.js:92:12)
    
    Request ID: 8f72a1b5-c3e4-42e1-9631-a3f892e1d504
    Client IP: 192.168.1.105
    Timestamp: 2025-06-18T14:32:15.234Z
    """
    
    log2 = """
    [2025-06-19 09:17:42] ERROR: JSON parsing failed
    File: /app/services/parser.js
    Line: 127
    Details: Unexpected token ] in JSON at position 892
    
    Stack trace:
    at JSON.parse (<anonymous>)
    at Parser.parseInput (/app/services/parser.js:127:23)
    at RequestHandler.processRequest (/app/handlers/request.js:45:18)
    at Server.handleConnection (/app/server.js:92:12)
    
    Request ID: 3e9a2c7d-f5b6-48d3-87a1-b4c5d6e7f8g9
    Client IP: 192.168.1.107
    Timestamp: 2025-06-19T09:17:42.567Z
    """
    
    # Test case 6: Scientific text with similar content but different terminology
    science1 = """
    Abstract: This study investigates the application of tree edit distance algorithms
    for comparing hierarchical data structures. We propose a novel approach that incorporates
    semantic understanding into the traditional tree edit distance calculation, allowing for
    more accurate comparison of structures with different naming conventions but similar meanings.
    
    Methods: We implemented a modified Zhang-Shasha algorithm with custom cost functions for
    insert, delete, and update operations. The algorithm was enhanced with embedding-based
    semantic similarity to recognize semantically equivalent node labels. We evaluated our
    approach on a dataset of 1,000 JSON documents from various domains.
    
    Results: Our semantic tree edit distance algorithm achieved a 27% improvement in accuracy
    compared to traditional syntactic approaches. The algorithm was particularly effective for
    documents with different naming conventions but equivalent structures, reducing false
    negatives by 42%.
    
    Conclusion: The integration of semantic understanding into tree edit distance calculations
    significantly improves the accuracy of hierarchical data comparison. This approach has
    applications in data integration, schema matching, and document similarity assessment.
    """
    
    science2 = """
    Abstract: In this paper, we examine how hierarchical data structures can be compared using
    modified tree distance algorithms. Our research introduces a semantic-aware methodology that
    enhances traditional tree comparison by incorporating meaning-based analysis, enabling more
    precise matching of structures that use different terminology for similar concepts.
    
    Methodology: A customized version of the tree edit distance algorithm was developed, featuring
    specialized cost functions for node operations (insertion, deletion, modification). We augmented
    this with vector embeddings to detect semantic equivalence between differently named nodes. The
    evaluation utilized 1,000 JSON documents spanning multiple domains.
    
    Findings: The semantically-enhanced tree distance algorithm demonstrated a 27% accuracy improvement
    over conventional syntax-based methods. Particularly noteworthy was its performance on structurally
    equivalent documents with terminology variations, where it reduced misclassifications by 42%.
    
    Discussion and Implications: By incorporating semantic analysis into hierarchical structure comparison,
    we achieve substantially more accurate similarity assessments. This technique offers valuable
    applications in data merging, schema alignment, and content similarity evaluation.
    """
    
    # Test case 7: Mixed content with code, text and structured data
    mixed1 = """
    # Project Documentation
    
    ## Data Structure
    
    The system uses the following JSON structure for configuration:
    
    ```json
    {
      "name": "ConfigurationSettings",
      "version": "1.0",
      "settings": {
        "timeout": 30,
        "retries": 3,
        "logging": {
          "level": "info",
          "format": "json"
        }
      }
    }
    ```
    
    ## Implementation
    
    The configuration is loaded using the following function:
    
    ```javascript
    function loadConfig(path) {
      const fs = require('fs');
      const data = fs.readFileSync(path, 'utf8');
      return JSON.parse(data);
    }
    ```
    
    ## Usage Guidelines
    
    Always validate the configuration before using it in production environments.
    The timeout should be adjusted based on network conditions, and logging levels
    should be set to "debug" only during development.
    """
    
    mixed2 = """
    # System Documentation
    
    ## Implementation Details
    
    Configuration loading is handled by this function:
    
    ```javascript
    function getConfiguration(configPath) {
      const fs = require('fs');
      const rawData = fs.readFileSync(configPath, 'utf8');
      return JSON.parse(rawData);
    }
    ```
    
    ## Configuration Format
    
    The configuration uses this JSON structure:
    
    ```json
    {
      "name": "SystemConfig",
      "version": "1.0",
      "settings": {
        "timeout": 30,
        "maxRetries": 3,
        "logging": {
          "level": "info",
          "outputFormat": "json"
        }
      }
    }
    ```
    
    ## Best Practices
    
    Make sure to validate all configuration values before using them.
    Adjust timeout settings based on your network performance, and only
    use debug logging in development environments.
    """
    
    # Run comparisons
    print("\nTest Case 1: Similar paragraphs with different wording")
    print(f"Standard comparison: {evaluator_standard._compare_strings(text1, text2):.4f}")
    print(f"Enhanced comparison: {evaluator_semantic._compare_strings(text1, text2):.4f}")
    
    print("\nTest Case 2: Similar code with different variable names")
    print(f"Standard comparison: {evaluator_standard._compare_strings(code1, code2):.4f}")
    print(f"Enhanced comparison: {evaluator_semantic._compare_strings(code1, code2):.4f}")
    
    print("\nTest Case 3: Partially similar text with added/removed sections")
    print(f"Standard comparison: {evaluator_standard._compare_strings(text3, text4):.4f}")
    print(f"Enhanced comparison: {evaluator_semantic._compare_strings(text3, text4):.4f}")
    
    print("\nTest Case 4: Technical documentation with reordered sections")
    print(f"Standard comparison: {evaluator_standard._compare_strings(doc1, doc2):.4f}")
    print(f"Enhanced comparison: {evaluator_semantic._compare_strings(doc1, doc2):.4f}")
    
    print("\nTest Case 5: Error logs with similar patterns but different details")
    print(f"Standard comparison: {evaluator_standard._compare_strings(log1, log2):.4f}")
    print(f"Enhanced comparison: {evaluator_semantic._compare_strings(log1, log2):.4f}")
    
    print("\nTest Case 6: Scientific text with similar content but different terminology")
    print(f"Standard comparison: {evaluator_standard._compare_strings(science1, science2):.4f}")
    print(f"Enhanced comparison: {evaluator_semantic._compare_strings(science1, science2):.4f}")
    
    print("\nTest Case 7: Mixed content with code, text and structured data")
    print(f"Standard comparison: {evaluator_standard._compare_strings(mixed1, mixed2):.4f}")
    print(f"Enhanced comparison: {evaluator_semantic._compare_strings(mixed1, mixed2):.4f}")
    
    # Test with complex JSON objects containing long text values
    json1 = {
        "metadata": {
            "title": "JSON Tree Consistency",
            "version": "1.0",
            "author": "Developer Team"
        },
        "documentation": {
            "overview": text1,
            "technical": doc1,
            "implementation": code1
        },
        "logs": {
            "recent": log1,
            "archived": False
        },
        "research": {
            "abstract": science1,
            "published": True,
            "citations": 42
        }
    }
    
    json2 = {
        "metadata": {
            "title": "JSON Structural Similarity",
            "version": "1.0.1",
            "author": "Development Team"
        },
        "documentation": {
            "overview": text2,
            "technical": doc2,
            "implementation": code2
        },
        "logs": {
            "recent": log2,
            "archived": False
        },
        "research": {
            "abstract": science2,
            "published": True,
            "citations": 45
        }
    }
    
    # Compare full JSON objects
    print("\nComparing complex JSON objects:")
    start_time = time.time()
    similarity, _ = evaluator_standard.calculate_tree_edit_distance(json1, json2)
    standard_time = time.time() - start_time
    print(f"Standard similarity: {similarity:.4f} (time: {standard_time:.2f}s)")
    
    start_time = time.time()
    similarity, _ = evaluator_semantic.calculate_tree_edit_distance(json1, json2)
    semantic_time = time.time() - start_time
    print(f"Semantic similarity: {similarity:.4f} (time: {semantic_time:.2f}s)")
    
    # Test with nested JSON containing mixed content types
    nested_json1 = {
        "project": {
            "name": "Data Analysis Tool",
            "description": "A tool for analyzing and visualizing data",
            "modules": [
                {
                    "name": "Parser",
                    "description": text3,
                    "implementation": code1,
                    "documentation": doc1
                },
                {
                    "name": "Analyzer",
                    "description": "Analyzes parsed data using statistical methods",
                    "implementation": "function analyze(data) { /* implementation */ }",
                    "documentation": "# Analyzer\n\nAnalyzes data using statistical methods."
                }
            ],
            "errors": [
                {
                    "id": "ERR001",
                    "description": "Parser error",
                    "details": log1
                }
            ]
        }
    }
    
    nested_json2 = {
        "project": {
            "name": "Data Visualization System",
            "description": "A system for analyzing and visualizing complex datasets",
            "modules": [
                {
                    "name": "DataParser",
                    "description": text4,
                    "implementation": code2,
                    "documentation": doc2
                },
                {
                    "name": "DataAnalyzer",
                    "description": "Performs statistical analysis on parsed datasets",
                    "implementation": "function performAnalysis(dataset) { /* implementation */ }",
                    "documentation": "# Data Analyzer\n\nPerforms statistical analysis on datasets."
                }
            ],
            "errors": [
                {
                    "id": "ERROR-001",
                    "description": "Parsing failure",
                    "details": log2
                }
            ]
        }
    }
    
    print("\nComparing nested JSON with mixed content types:")
    start_time = time.time()
    similarity, _ = evaluator_standard.calculate_tree_edit_distance(nested_json1, nested_json2)
    standard_time = time.time() - start_time
    print(f"Standard similarity: {similarity:.4f} (time: {standard_time:.2f}s)")
    
    start_time = time.time()
    similarity, _ = evaluator_semantic.calculate_tree_edit_distance(nested_json1, nested_json2)
    semantic_time = time.time() - start_time
    print(f"Semantic similarity: {similarity:.4f} (time: {semantic_time:.2f}s)")

if __name__ == "__main__":
    print_separator()
    print("ENHANCED LONG TEXT COMPARISON TEST")
    print_separator()
    
    try:
        test_long_text_comparison()
    except Exception as e:
        import traceback
        print(f"Error during testing: {e}")
        traceback.print_exc()
    
    print_separator()
    print("Test completed!")
    print_separator()