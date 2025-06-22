import json
import numpy as np
from typing import List, Dict, Any, Union, Tuple, Optional, Set
from collections import defaultdict
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
import warnings
import asyncio
import concurrent.futures
from functools import partial
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from cal_similarity_structured_data import SemanticJSONSimilarity, EMBEDDINGS_AVAILABLE

class EmbeddingBasedConsistencyAnalyzer(SemanticJSONSimilarity):
    def __init__(self, 
                 semantic_similarity_config: Optional[Dict[str, Any]] = None,
                 use_multiple_analyzers: bool = True,
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize consistency analyzer with embedding-based semantic similarity.
        
        Args:
            semantic_similarity_config: Configuration for semantic similarity calculator
            use_multiple_analyzers: Whether to use multiple similarity configurations
            embedding_model: Name of the sentence transformer model to use
        """
        # Default configuration
        default_config = {
            'structure_weight': 0.5,
            'value_weight': 0.5,
            'array_order_matters': False,
            'number_tolerance': 0.01,
            'string_method': 'semantic',  # Use semantic for strings too
            'use_semantic_similarity': True,
            'embedding_model': embedding_model,
            'semantic_threshold': 0.7,
            'key_semantic_weight': 0.8,
            'exact_match_weight': 0.2
        }
        
        if semantic_similarity_config:
            default_config.update(semantic_similarity_config)
        
        # Initialize the parent class (SemanticJSONSimilarity)
        super().__init__(**default_config)
        
        # Additional analyzers for comparison
        self.analyzers = {}
        if use_multiple_analyzers and EMBEDDINGS_AVAILABLE:
            # Strict analyzer (no semantic matching)
            self.analyzers['strict'] = SemanticJSONSimilarity(
                structure_weight=0.5,
                value_weight=0.5,
                use_semantic_similarity=False,
                string_method='exact',
                embedding_model=embedding_model
            )
            
            # Structure-focused analyzer with embeddings
            self.analyzers['structure_focused'] = SemanticJSONSimilarity(
                structure_weight=0.8,
                value_weight=0.2,
                use_semantic_similarity=True,
                semantic_threshold=0.6,
                embedding_model=embedding_model
            )
            
            # Value-focused analyzer with semantic string matching
            self.analyzers['value_focused'] = SemanticJSONSimilarity(
                structure_weight=0.2,
                value_weight=0.8,
                use_semantic_similarity=True,
                string_method='semantic',
                embedding_model=embedding_model
            )
            
            # High-threshold analyzer (very strict semantic matching)
            self.analyzers['high_threshold'] = SemanticJSONSimilarity(
                structure_weight=0.5,
                value_weight=0.5,
                use_semantic_similarity=True,
                semantic_threshold=0.9,
                key_semantic_weight=0.9,
                embedding_model=embedding_model
            )
        
        self.results_history = []
        
    def _create_consistency_report(self, field_consistency_results: Dict[str, Dict[str, Any]], valid_outputs: List[dict]) -> Dict[str, Any]:
        """
        Create a detailed consistency report from field consistency results.
        
        Args:
            field_consistency_results: Dictionary mapping field names to consistency metrics
            valid_outputs: List of valid outputs that were analyzed
            
        Returns:
            Dictionary containing the full consistency report
        """
        try:
            # Calculate aggregate metrics
            all_scores = [metrics.get('overall_consistency', 0.0) 
                        for metrics in field_consistency_results.values()]
            
            # Identify problematic fields
            problematic_fields = [
                field for field, metrics in field_consistency_results.items()
                if metrics.get('overall_consistency', 0.0) < 0.7
            ]
            
            # Create final report
            report = {
                'timestamp': datetime.now().isoformat(),
                'num_outputs_analyzed': len(valid_outputs),
                'num_unique_field_groups': len(field_consistency_results),
                'aggregate_metrics': {
                    'mean_field_consistency': float(np.mean(all_scores)) if all_scores else 0.0,
                    'std_field_consistency': float(np.std(all_scores)) if all_scores else 0.0,
                    'min_field_consistency': float(np.min(all_scores)) if all_scores else 0.0,
                    'max_field_consistency': float(np.max(all_scores)) if all_scores else 1.0,
                },
                'field_consistency_scores': field_consistency_results,
                'problematic_fields': problematic_fields,
                'most_consistent_fields': self._get_top_fields(field_consistency_results, n=5, ascending=False) if field_consistency_results else [],
                'least_consistent_fields': self._get_top_fields(field_consistency_results, n=5, ascending=True) if field_consistency_results else [],
                #'recommendations': self._generate_field_recommendations(field_consistency_results) if field_consistency_results else []
            }
            
            return report
        except Exception as e:
            print(f"Error generating field consistency report: {e}")
            # Return a minimal valid report
            return {
                'timestamp': datetime.now().isoformat(),
                'error': f"Failed to generate complete report: {str(e)}",
                'num_outputs_analyzed': len(valid_outputs),
                'num_unique_field_groups': len(field_consistency_results) if field_consistency_results else 0,
                'aggregate_metrics': {},
                'field_consistency_scores': {},
                'problematic_fields': [],
                'most_consistent_fields': [],
                'least_consistent_fields': [],
                'recommendations': ["Unable to generate recommendations due to an error."]
            }
    
    def _filter_fields_by_presence(self, field_consistency_results: Dict[str, Dict[str, Any]], min_presence_ratio: float = 0.5) -> Dict[str, Dict[str, Any]]:
        """
        Filter fields based on minimum presence ratio and format field names.
        
        Args:
            field_consistency_results: Dictionary mapping field names to consistency metrics
            min_presence_ratio: Minimum ratio of outputs that must have a field for it to be included
            
        Returns:
            Filtered dictionary with formatted field names
        """
        filtered_results = {}
        
        for field_name, metrics in field_consistency_results.items():
            # Check if field appears in enough outputs
            presence_ratio = metrics.get('presence_ratio', 0.0)
            
            if presence_ratio >= min_presence_ratio:
                # Add variation count to field name if applicable
                display_name = field_name
                variations = metrics.get('key_variations', [])
                if len(variations) > 1:
                    display_name = f"{field_name} ({len(variations)} variations)"
                
                filtered_results[display_name] = metrics
        
        return filtered_results
    
    def get_field_consistency_report(self, 
                               outputs: List[Union[str, dict]], 
                               include_nested: bool = True,
                               min_presence_ratio: float = 0.5) -> Dict[str, Any]:
        """
        Generate a detailed field-level consistency report.
        
        Args:
            outputs: List of JSON outputs to analyze
            include_nested: Whether to include nested field paths
            min_presence_ratio: Minimum ratio of outputs that must have a field for it to be included
            
        Returns:
            Dictionary containing field-level consistency metrics
        """
        try:
            # Parse outputs
            parsed_outputs = self._parse_outputs(outputs)
            valid_outputs = [o for o in parsed_outputs if o is not None]
            
            if len(valid_outputs) < 2:
                return {
                    "error": "Not enough valid outputs for field consistency analysis",
                    "valid_count": len(valid_outputs)
                }
            
            # Step 1: Analyze key consistency across responses using semantic similarity
            field_consistency_results = self._analyze_key_consistency_across_responses(valid_outputs)
            
            # Step 2: Filter fields based on minimum presence ratio
            filtered_results = self._filter_fields_by_presence(field_consistency_results, min_presence_ratio)
            
            # Step 3: Create the full consistency report
            return self._create_consistency_report(filtered_results, valid_outputs)
        except Exception as e:
            print(f"Error in get_field_consistency_report: {e}")
            return {
                "error": f"Failed to analyze field consistency: {str(e)}",
                "valid_count": len(outputs) if outputs else 0
            }
        
    async def get_field_consistency_report_async(self, 
                                      outputs: List[Union[str, dict]], 
                                      include_nested: bool = True,
                                      min_presence_ratio: float = 0.5) -> Dict[str, Any]:
        """
        Generate a detailed field-level consistency report asynchronously.
        
        Args:
            outputs: List of JSON outputs to analyze
            include_nested: Whether to include nested field paths
            min_presence_ratio: Minimum ratio of outputs that must have a field for it to be included
            
        Returns:
            Dictionary containing field-level consistency metrics
        """
        try:
            # Parse outputs
            parsed_outputs = self._parse_outputs(outputs)
            valid_outputs = [o for o in parsed_outputs if o is not None]
            
            if len(valid_outputs) < 2:
                return {
                    "error": "Not enough valid outputs for field consistency analysis",
                    "valid_count": len(valid_outputs)
                }
            
            # Step 1: Analyze key consistency across responses using semantic similarity (async version)
            field_consistency_results = await self._analyze_key_consistency_across_responses_async(valid_outputs)
            
            # Step 2: Filter fields based on minimum presence ratio (reuse the helper method)
            filtered_results = self._filter_fields_by_presence(field_consistency_results, min_presence_ratio)
            
            # Step 3: Create the full consistency report (reuse the helper method)
            return self._create_consistency_report(filtered_results, valid_outputs)
        except Exception as e:
            print(f"Error in get_field_consistency_report_async: {e}")
            return {
                "error": f"Failed to analyze field consistency asynchronously: {str(e)}",
                "valid_count": len(outputs) if outputs else 0
            }

    def _find_most_common_values(self, values: List[Any]) -> List[Dict[str, Any]]:
        """Find the most common values with semantic clustering.
        
        Args:
            values: List of values to analyze
            
        Returns:
            List of dictionaries with common values and their frequencies
        """
        if not values:
            return []
        
        # For string values, we need to handle semantic similarity
        if all(isinstance(val, str) for val in values):
            # Group semantically similar values
            clusters = []
            
            for value in values:
                # Check if this value belongs to an existing cluster
                assigned = False
                for cluster in clusters:
                    # Compare with the representative of the cluster
                    similarity = self._calculate_semantic_similarity(value, cluster['representative'])
                    if similarity >= self.semantic_threshold:
                        cluster['values'].append(value)
                        cluster['count'] += 1
                        assigned = True
                        break
                
                # If not assigned to any existing cluster, create a new one
                if not assigned:
                    clusters.append({
                        'representative': value,
                        'values': [value],
                        'count': 1
                    })
            
            # Sort clusters by count in descending order
            sorted_clusters = sorted(clusters, key=lambda x: x['count'], reverse=True)
            
            # Return the most common clusters with their frequencies
            return [{
                'value': cluster['representative'],
                'count': cluster['count'],
                'variations': cluster['values'][:5]  # Limit to 5 variations
            } for cluster in sorted_clusters]
        
        # For non-string values, handle different types appropriately
        else:
            from collections import defaultdict
            value_counts = defaultdict(int)
            
            # Count occurrences manually to handle unhashable types like dictionaries
            for value in values:
                # For dictionaries, convert to a string representation for counting
                if isinstance(value, dict):
                    # Use a stable string representation (sorted keys)
                    value_str = str(sorted(value.items()))
                    value_counts[value_str] += 1
                # For lists, convert to tuple for hashability
                elif isinstance(value, list):
                    try:
                        value_tuple = tuple(value)
                        value_counts[value_tuple] += 1
                    except TypeError:  # Contains unhashable types
                        value_str = str(value)
                        value_counts[value_str] += 1
                # For hashable types, count directly
                else:
                    try:
                        value_counts[value] += 1
                    except TypeError:  # Unhashable type
                        value_str = str(value)
                        value_counts[value_str] += 1
            
            # Convert back to original values and sort by count
            result = []
            for value_key, count in sorted(value_counts.items(), key=lambda x: x[1], reverse=True):
                # If it's a string representation of a dict or list, use the original value
                original_value = None
                for v in values:
                    if isinstance(v, dict) and str(sorted(v.items())) == value_key:
                        original_value = v
                        break
                    elif isinstance(v, list) and str(v) == value_key:
                        original_value = v
                        break
                
                # If we didn't find an original value, use the key directly
                if original_value is None:
                    original_value = value_key
                
                result.append({
                    'value': original_value,
                    'count': count,
                    'variations': [original_value]
                })
            
            return result

    def _get_top_fields(self, field_results: Dict[str, Dict[str, Any]], n: int = 5, ascending: bool = True) -> List[Dict[str, Any]]:
        """Get top N fields by consistency score."""
        if not field_results:
            return []
            
        try:
            # Sort fields by consistency score
            sorted_fields = sorted(
                field_results.items(),
                key=lambda x: x[1].get('overall_consistency', x[1].get('consistency_score', 0.0)),
                reverse=not ascending
            )
            
            result = []
            for field, metrics in sorted_fields[:n]:
                # Handle both old and new field names for backward compatibility
                presence = metrics.get('presence_ratio', metrics.get('occurrence_ratio', 0.0))
                consistency = metrics.get('overall_consistency', metrics.get('consistency_score', 0.0))
                
                result.append({
                    'field': field,
                    'consistency_score': consistency,
                    'presence_ratio': presence,
                    'key_variations': metrics.get('key_variations', [])
                })
                
            return result
        except Exception as e:
            print(f"Error getting top fields: {e}")
            return []
    
    def analyze_embedding_consistency(self, 
                                    outputs: List[Union[str, dict]], 
                                    prompt: str = None,
                                    model_params: dict = None,
                                    detailed_analysis: bool = True,
                                    analyze_embeddings: bool = True) -> Dict[str, Any]:
        """
        Analyze consistency using embedding-based semantic understanding.
        
        Args:
            outputs: List of JSON outputs
            prompt: The prompt used
            model_params: Model parameters
            detailed_analysis: Whether to perform detailed semantic analysis
            analyze_embeddings: Whether to analyze embedding space relationships
            
        Returns:
            Comprehensive embedding-based consistency analysis
        """
        # Parse outputs
        parsed_outputs = self._parse_outputs(outputs)
        valid_outputs = [o for o in parsed_outputs if o is not None]
        
        if len(valid_outputs) < 2:
            return {
                "error": "Not enough valid JSON outputs to analyze",
                "valid_count": len(valid_outputs),
                "total_count": len(outputs)
            }
        
        # Calculate similarity matrices with different analyzers
        n = len(valid_outputs)
        similarity_matrices = {}
        
        # Main semantic similarity matrix using parent class's compare method
        semantic_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    semantic_matrix[i][j] = self.compare(
                        valid_outputs[i], valid_outputs[j]
                    )
                else:
                    semantic_matrix[i][j] = 1.0
        
        similarity_matrices['semantic'] = semantic_matrix
        
        # Calculate other similarity types
        if self.analyzers:
            for analyzer_name, analyzer in self.analyzers.items():
                matrix = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            matrix[i][j] = analyzer.compare(valid_outputs[i], valid_outputs[j])
                        else:
                            matrix[i][j] = 1.0
                similarity_matrices[analyzer_name] = matrix
        
        # Perform improved embedding analysis if requested
        embedding_analysis = None
        cross_response_analysis = None
        structural_analysis = None
        type_consistency_analysis = None
        
        if analyze_embeddings and self.embedding_model and EMBEDDINGS_AVAILABLE:
            # New improved analysis
            embedding_analysis = self._analyze_embedding_space(valid_outputs)
            
            # Extract specific analyses for easier access
            if embedding_analysis:
                cross_response_analysis = embedding_analysis.get("cross_response_key_consistency")
                structural_analysis = embedding_analysis.get("structural_similarity")
                type_consistency_analysis = embedding_analysis.get("type_consistency")
        
        # Identify semantic clusters using embeddings
        semantic_clusters = self._identify_embedding_clusters(semantic_matrix, valid_outputs)
        
        # Calculate advanced metrics
        metrics = self._calculate_embedding_metrics(
            similarity_matrices,
            valid_outputs,
            embedding_analysis
        )
        
        # Perform detailed comparison analysis
        detailed_comparisons = None
        if detailed_analysis:
            detailed_comparisons = self._perform_detailed_embedding_analysis(valid_outputs)
        
        # Create comprehensive result
        result = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "model_params": model_params,
            "num_outputs": len(outputs),
            "num_valid_outputs": len(valid_outputs),
            "parsing_success_rate": len(valid_outputs) / len(outputs),
            "embedding_model_used": self.embedding_model.__class__.__name__ if self.embedding_model else "None",
            "metrics": metrics,
            "semantic_clusters": semantic_clusters,
            "embedding_analysis": embedding_analysis,
            "cross_response_analysis": cross_response_analysis,
            "structural_analysis": structural_analysis,
            "type_consistency_analysis": type_consistency_analysis,
            "similarity_matrices": {k: v.tolist() for k, v in similarity_matrices.items()},
            "detailed_comparisons": detailed_comparisons,
            "outputs": valid_outputs
        }
        
        # Store for historical analysis
        self.results_history.append(result)
        
        return result
    
    def _parse_outputs(self, outputs: List[Union[str, dict]]) -> List[Optional[dict]]:
        """Parse JSON outputs safely."""
        parsed = []
        for output in outputs:
            if isinstance(output, str):
                try:
                    parsed.append(json.loads(output))
                except json.JSONDecodeError:
                    parsed.append(None)
            else:
                parsed.append(output)
        return parsed
    
    def _get_value_by_path(self, obj: dict, path: str) -> Any:
        """Get a value from a nested dictionary using a dot-separated path."""
        if not obj or not path:
            return None
            
        # If path is already a list of parts, use it directly
        if isinstance(path, list):
            parts = path
        else:
            # If path contains dots, it's a nested path
            parts = path.split('.')
            
        current = obj
        
        for part in parts:
            # Handle array notation with []  
            if part.endswith('[]'):
                array_part = part[:-2]  # Remove [] suffix
                if array_part in current and isinstance(current[array_part], list):
                    # Return the whole array for [] notation
                    current = current[array_part]
                else:
                    return None
            # Handle array indexing with [0], [1], etc.
            elif '[' in part and ']' in part:
                array_part = part.split('[')[0]
                index_str = part.split('[')[1].split(']')[0]
                try:
                    index = int(index_str)
                    if array_part in current and isinstance(current[array_part], list) and len(current[array_part]) > index:
                        current = current[array_part][index]
                    else:
                        return None
                except (ValueError, KeyError):
                    return None
            else:
                # Regular key access
                if isinstance(current, dict) and part in current:
                    current = current[part]
                elif isinstance(current, list) and part.isdigit():
                    # Handle list indices as strings
                    try:
                        current = current[int(part)]
                    except (IndexError, ValueError):
                        return None
                else:
                    return None
        
        return current
        
    def _set_nested_value(self, dictionary: dict, path: str, value: Any) -> None:
        """Set value in nested dictionary using dot notation."""
        if not path:
            return
            
        # If path is already a list of parts, use it directly
        if isinstance(path, list):
            parts = path
        else:
            # If path contains dots, it's a nested path
            parts = path.split('.')
            
        current = dictionary
        
        # Navigate to the parent of the final key
        for i, part in enumerate(parts[:-1]):
            # Handle array notation
            if part.endswith('[]'):
                part = part[:-2]  # Remove [] suffix
                
            # Handle array indexing
            if '[' in part and ']' in part:
                array_part = part.split('[')[0]
                index_str = part.split('[')[1].split(']')[0]
                
                # Ensure the array exists
                if array_part not in current:
                    current[array_part] = []
                    
                # Ensure the array is long enough
                try:
                    index = int(index_str)
                    while len(current[array_part]) <= index:
                        current[array_part].append({})  # Extend the array
                    current = current[array_part][index]
                except ValueError:
                    return  # Invalid index
            else:
                # Regular key access
                if part not in current:
                    current[part] = {}
                current = current[part]
        
        # Set the final value
        final_part = parts[-1]
        
        # Handle array notation for final part
        if final_part.endswith('[]'):
            final_part = final_part[:-2]  # Remove [] suffix
            if final_part not in current:
                current[final_part] = []
            current[final_part].append(value)
        elif '[' in final_part and ']' in final_part:
            array_part = final_part.split('[')[0]
            index_str = final_part.split('[')[1].split(']')[0]
            
            # Ensure the array exists
            if array_part not in current:
                current[array_part] = []
                
            # Set the value at the specified index
            try:
                index = int(index_str)
                while len(current[array_part]) <= index:
                    current[array_part].append(None)  # Extend the array
                current[array_part][index] = value
            except ValueError:
                return  # Invalid index
        else:
            # Regular key assignment
            current[final_part] = value
    
    # _extract_structure method removed - using parent class's functionality instead
    
    # _compare_structures method removed - using parent class's compare method instead
    
    def _extract_and_group_keys(self, outputs: List[dict]) -> Tuple[List[Tuple[int, str]], Dict[str, Dict[str, Any]], int]:
        """Extract and group keys from outputs.
        
        Args:
            outputs: List of output dictionaries
            
        Returns:
            Tuple containing:
            - List of (output_idx, key) tuples
            - Dictionary of semantic key groups
            - Total number of responses
        """
        # Extract all keys from each output with their output index
        output_keys_with_idx = []
        for output_idx, output in enumerate(outputs):
            keys = self._flatten_keys(output, include_nested=True)
            for key in keys:
                output_keys_with_idx.append((output_idx, key))
        
        total_responses = len(outputs)
        
        # Group semantically similar keys
        semantic_key_groups = self._group_semantically_similar_keys([k for _, k in output_keys_with_idx])
        
        return output_keys_with_idx, semantic_key_groups, total_responses
    
    def _collect_occurrences(self, outputs: List[dict], key_variations: Set[str]) -> List[Tuple[int, Any]]:
        """Collect occurrences of values for a set of key variations across outputs.
        
        Args:
            outputs: List of output dictionaries
            key_variations: Set of key variations to look for
            
        Returns:
            List of (output_idx, value) tuples
        """
        occurrences = []
        for output_idx, output in enumerate(outputs):
            # Try each key variation to find a value
            for key in key_variations:
                value = self._get_value_by_path(output, key)
                if value is not None:
                    occurrences.append((output_idx, value))
                    break  # Found a value for this output, move to next output
        return occurrences
    
    def _analyze_key_consistency_across_responses(self, outputs: List[dict]) -> Dict[str, Any]:
        """Analyze consistency of keys across different responses using semantic similarity."""
        try:
            # Extract and group keys
            _, semantic_key_groups, total_responses = self._extract_and_group_keys(outputs)
            
            # For each semantic key group, collect values across responses
            key_consistency = {}
            
            for group_id, group_info in semantic_key_groups.items():
                representative_key = group_info['representative_key']
                key_variations = group_info['keys']
                
                # Collect values for all keys in this semantic group
                occurrences = self._collect_occurrences(outputs, key_variations)
                
                # Calculate consistency metrics for this semantic key group
                if occurrences:  # Only analyze if we found any occurrences
                    key_consistency[representative_key] = self._calculate_key_consistency(
                        representative_key, occurrences, total_responses=total_responses
                    )
                    
                    # Add information about key variations
                    key_consistency[representative_key]['key_variations'] = list(key_variations)
                    key_consistency[representative_key]['semantic_key_group'] = True
            
            return key_consistency
        except Exception as e:
            print(f"Error in _analyze_key_consistency_across_responses: {e}")
            return {}
        
    async def _analyze_key_consistency_across_responses_async(self, outputs: List[dict]) -> Dict[str, Any]:
        """Analyze consistency of keys across different responses using semantic similarity (async version)."""
        # Extract all keys from each output with their output index
        output_keys_with_idx = []
        for output_idx, output in enumerate(outputs):
            keys = self._flatten_keys(output, include_nested=True)
            for key in keys:
                output_keys_with_idx.append((output_idx, key))
        
        total_responses = len(outputs)
        
        # Group semantically similar keys
        semantic_key_groups = self._group_semantically_similar_keys([k for _, k in output_keys_with_idx])
        
        # For each semantic key group, collect values across responses
        key_consistency = {}
        
        # Process groups concurrently
        tasks = []
        for group_id, group_info in semantic_key_groups.items():
            representative_key = group_info['representative_key']
            key_variations = group_info['keys']
            
            # Collect values for all keys in this semantic group
            occurrences = self._collect_occurrences(outputs, key_variations)
            
            # Only analyze if we found any occurrences
            if occurrences:
                # Create a task for calculating consistency metrics
                tasks.append((representative_key, key_variations, self._calculate_key_consistency_async(
                    representative_key, occurrences, total_responses=total_responses
                )))
        
        # Await all tasks
        for representative_key, key_variations, task_coro in tasks:
            metrics = await task_coro
            key_consistency[representative_key] = metrics
            
            # Add information about key variations
            key_consistency[representative_key]['key_variations'] = list(key_variations)
            key_consistency[representative_key]['semantic_key_group'] = True
        
        return key_consistency
        
    async def _calculate_key_consistency_async(self, key: str, occurrences: List[Tuple[int, Any]], total_responses: int = None) -> Dict[str, Any]:
        """Calculate consistency metrics for a specific key across responses (async version)."""
        # This is the async version of _calculate_key_consistency
        # For now, we'll just call the synchronous version, but this could be optimized later
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor,
                lambda: self._calculate_key_consistency(key, occurrences, total_responses)
            )
        
    def _group_semantically_similar_keys(self, all_keys: List[str]) -> Dict[str, Dict[str, Any]]:
        """Group semantically similar keys together while respecting hierarchical structure."""
        if not all_keys:
            return {}
            
        # Group keys by their hierarchical level first
        level_grouped_keys = {}
        for key in all_keys:
            # Count dots to determine hierarchy level
            level = key.count('.')
            if level not in level_grouped_keys:
                level_grouped_keys[level] = []
            level_grouped_keys[level].append(key)
        
        # Create groups of semantically similar keys, only comparing within the same level
        key_groups = {}
        group_counter = 0
        processed_keys = set()
        
        # Process each hierarchical level separately
        for level, keys_at_level in level_grouped_keys.items():
            for key1 in keys_at_level:
                if key1 in processed_keys:
                    continue
                    
                # Create new group
                group_id = f"group_{group_counter}"
                key_groups[group_id] = {
                    'representative_key': key1,
                    'keys': {key1},
                    'hierarchy_level': level
                }
                processed_keys.add(key1)
                
                # Find semantically similar keys at the same hierarchical level
                for key2 in keys_at_level:
                    if key2 in processed_keys:
                        continue
                    
                    # Check if keys are in the same hierarchy branch
                    if self._are_keys_in_same_branch(key1, key2):
                        continue  # Skip keys in the same branch but at different levels
                    
                    # Calculate semantic similarity between keys
                    similarity = self._calculate_key_similarity(key1, key2)
                    
                    # If similarity exceeds threshold, add to group
                    if similarity >= self.semantic_threshold:
                        key_groups[group_id]['keys'].add(key2)
                        processed_keys.add(key2)
                
                group_counter += 1
        
        return key_groups
        
    def _are_keys_in_same_branch(self, key1: str, key2: str) -> bool:
        """Check if two keys are in the same hierarchical branch.
        
        For example, 'homeAddr' and 'homeAddr.streetName' are in the same branch,
        while 'homeAddr' and 'workAddr' are not.
        """
        # If one key is a prefix of the other (followed by a dot), they're in the same branch
        if key1.startswith(key2 + '.') or key2.startswith(key1 + '.'):
            return True
            
        # Handle array notation
        key1_base = key1.split('[')[0]  # Remove array indices
        key2_base = key2.split('[')[0]
        
        if key1_base.startswith(key2_base + '.') or key2_base.startswith(key1_base + '.'):
            return True
            
        return False
    
    def _calculate_set_similarity(self, set1: List[Any], set2: List[Any], return_pairs: bool = False, filter_threshold: float = 0.5) -> Union[float, Tuple[float, List[Tuple[Any, Any, float]]]]:
        """Calculate similarity between two sets of elements using maximum bipartite matching.
        
        Args:
            set1, set2: Lists of elements to compare
            return_pairs: Whether to return matched pairs along with similarity score
            filter_threshold: Minimum similarity threshold for including pairs in results
            
        Returns:
            Similarity score between 0 and 1, and optionally matched pairs
        """
        # Handle empty sets
        if not set1 and not set2:
            return 1.0  # Two empty sets are identical
        
        if not set1 or not set2:
            # Empty set vs non-empty is always 0
            return 0.0
        
        # Create similarity matrix
        similarity_matrix = np.zeros((len(set1), len(set2)))
        for i, elem1 in enumerate(set1):
            for j, elem2 in enumerate(set2):
                similarity_matrix[i, j] = self._compare_nodes(elem1, elem2)
        
        # Use Hungarian algorithm to find maximum weight matching
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)  # Negate for max weight
        
        # Calculate total similarity
        total_similarity = sum(similarity_matrix[row_ind, col_ind])
        
        # Normalize by the size of the larger set
        normalized_similarity = total_similarity / max(len(set1), len(set2))
        
        if return_pairs:
            # Create pairs of matched elements with their similarity scores
            # Only include pairs that meet the threshold
            matched_pairs = [
                (set1[i], set2[j], similarity_matrix[i, j]) 
                for i, j in zip(row_ind, col_ind)
                if similarity_matrix[i, j] >= filter_threshold
            ]
            return normalized_similarity, matched_pairs
        
        return normalized_similarity
    
    def _calculate_key_consistency(self, key: str, occurrences: List[Tuple[int, Any]], total_responses: int = None) -> Dict[str, Any]:
        """Calculate consistency metrics for a specific key across responses.
        
        Args:
            key: The field/key name
            occurrences: List of (output_idx, value) tuples for this key
            total_responses: Total number of responses being analyzed
            
        Returns:
            Dictionary with consistency metrics
        """
        values = [value for _, value in occurrences]
        occurrence_count = len(values)
        
        # Get value types
        value_types = list(set(type(v).__name__ for v in values))
        type_consistency = len(value_types) == 1
        
        # Calculate presence ratio (how often this field appears)
        presence_ratio = 1.0
        if total_responses is not None and total_responses > 0:
            presence_ratio = occurrence_count / total_responses
        
        # Calculate value similarity based on type
        if len(values) < 2:
            # Even with one occurrence, consistency should reflect presence ratio
            return {
                "occurrence_count": occurrence_count,
                "presence_ratio": presence_ratio,
                "type_consistency": True,
                "value_types": value_types,
                "value_similarity": 1.0,
                "overall_consistency": presence_ratio,  # Scale by presence
                "sample_values": values[:3]  # First 3 samples
            }
        
        # For list values, calculate set similarity using Hungarian algorithm
        if any(isinstance(v, list) for v in values):
            # Group values by output index
            output_to_values = {}
            for output_idx, value in occurrences:
                if output_idx not in output_to_values:
                    output_to_values[output_idx] = []
                if isinstance(value, list):
                    output_to_values[output_idx].extend(value)
                else:
                    output_to_values[output_idx].append(value)
            
            # Calculate pairwise set similarities
            similarities = []
            output_indices = list(output_to_values.keys())
            for i in range(len(output_indices) - 1):
                for j in range(i + 1, len(output_indices)):
                    set1 = output_to_values[output_indices[i]]
                    set2 = output_to_values[output_indices[j]]
                    sim = self._calculate_set_similarity(set1, set2)
                    similarities.append(sim)
        else:
            # Calculate pairwise similarities using parent class method
            similarities = []
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    # Use parent class method directly
                    sim = self._compare_nodes(values[i], values[j])
                    similarities.append(sim)
        
        mean_similarity = float(np.mean(similarities)) if similarities else 1.0
        similarity_std = float(np.std(similarities)) if len(similarities) > 1 else 0.0
        
        # Calculate overall consistency score that accounts for:
        # 1. Presence ratio (how often the field appears)
        # 2. Value similarity (how consistent the values are when present)
        # 3. Type consistency (whether the field has consistent types)
        
        # Value consistency component (mean similarity adjusted by standard deviation)
        value_consistency = mean_similarity * (1.0 - 0.5 * similarity_std)
        
        # Type consistency component (binary factor)
        type_factor = 1.0 if type_consistency else 0.8
        
        # Overall consistency score
        overall_consistency = presence_ratio * value_consistency * type_factor
        
        # Find most common values and their variations
        try:
            common_values = self._find_most_common_values(values)
        except Exception as e:
            print(f"Warning: Error finding common values: {e}")
            common_values = []
        
        return {
            "occurrence_count": occurrence_count,
            "presence_ratio": presence_ratio,
            "type_consistency": type_consistency,
            "value_types": value_types,
            "value_similarity": mean_similarity,
            "value_similarity_std": similarity_std,
            "overall_consistency": overall_consistency,
            "sample_values": values[:3] if len(values) >= 3 else values,  # First 3 samples or all if fewer
            "common_values": common_values[:5] if common_values else []
        }
    
    def _analyze_structural_similarity(self, outputs: List[dict]) -> Dict[str, Any]:
        """Analyze the structural similarity of responses using parent class comparison."""
        # Calculate pairwise structural similarity using parent class compare method
        similarities = []
        different_pairs = []
        
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                # Use the parent class comparison method
                sim = self.compare(outputs[i], outputs[j])
                similarities.append(sim)
                
                # Record pairs with low similarity
                if sim < 0.8:  # Threshold for "different" structures
                    different_pairs.append((i, j, sim))
        
        mean_similarity = float(np.mean(similarities)) if similarities else 1.0
        
        return {
            "mean_structural_similarity": mean_similarity,
            "std_structural_similarity": float(np.std(similarities)) if len(similarities) > 1 else 0.0,
            "min_structural_similarity": float(np.min(similarities)) if similarities else 1.0,
            "max_structural_similarity": float(np.max(similarities)) if similarities else 1.0,
            "different_structure_pairs": different_pairs,
            "perfect_structural_consistency": all(sim > 0.99 for sim in similarities) if similarities else True
        }
    
    def _analyze_value_type_consistency(self, outputs: List[dict]) -> Dict[str, Any]:
        """Analyze consistency of value types for each field across responses."""
        # Extract all keys from each output
        output_keys = [self._flatten_keys(output) for output in outputs]
        all_keys = set().union(*output_keys) if output_keys else set()
        
        # For each key, check value type consistency
        type_consistency = {}
        inconsistent_fields = []
        
        for key in all_keys:
            types = []
            values_by_type = defaultdict(list)
            
            for output_idx, output in enumerate(outputs):
                value = self._get_value_by_path(output, key)
                if value is not None:
                    value_type = type(value).__name__
                    types.append(value_type)
                    values_by_type[value_type].append((output_idx, value))
            
            if len(types) > 1:
                # Calculate type consistency
                is_consistent = len(set(types)) == 1
                type_consistency[key] = {
                    "is_consistent": is_consistent,
                    "types": list(set(types)),
                    "type_distribution": {t: types.count(t) for t in set(types)},
                    "sample_values": {t: values_by_type[t][:2] for t in values_by_type}
                }
                
                if not is_consistent:
                    inconsistent_fields.append(key)
        
        return {
            "field_type_consistency": type_consistency,
            "inconsistent_fields": inconsistent_fields,
            "consistency_ratio": 1.0 - (len(inconsistent_fields) / len(all_keys) if all_keys else 0)
        }
    
    def _analyze_embedding_space(self, outputs: List[dict]) -> Dict[str, Any]:
        """Analyze outputs in the embedding space with improved context awareness."""
        if not self.embedding_model:
            return None
        
        # Perform cross-response consistency analysis with error handling
        try:
            key_consistency = self._analyze_key_consistency_across_responses(outputs)
        except Exception as e:
            print(f"Warning: Error in key consistency analysis: {e}")
            key_consistency = {}
            
        try:
            structural_similarity = self._analyze_structural_similarity(outputs)
        except Exception as e:
            print(f"Warning: Error in structural similarity analysis: {e}")
            structural_similarity = {}
            
        try:
            type_consistency = self._analyze_value_type_consistency(outputs)
        except Exception as e:
            print(f"Warning: Error in type consistency analysis: {e}")
            type_consistency = {}
        
        print(f"key_consistency: {key_consistency}")
        print(f"structural_similarity: {structural_similarity}")
        print(f"type_consistency: {type_consistency}")
        
        # Collect field-specific embeddings (grouped by field path)
        field_embeddings = {}
        try:
            for output in outputs:
                flat_values = self._extract_field_values(output)
                for path, value in flat_values:
                    if isinstance(value, str) and len(value) > 0:
                        if path not in field_embeddings:
                            field_embeddings[path] = []
                        field_embeddings[path].append(value)
        except Exception as e:
            print(f"Warning: Error extracting field values: {e}")
        
        # Calculate field-specific embedding diversity
        field_diversity = {}
        from sklearn.metrics.pairwise import cosine_distances
        
        for field, values in field_embeddings.items():
            try:
                if len(values) > 1:
                    # Get embeddings for all values of this field
                    embeddings = []
                    for value in values:
                        try:
                            embedding = self._get_embedding(value)
                            if embedding is not None:
                                embeddings.append(embedding)
                        except Exception as e:
                            print(f"Warning: Error getting embedding for value: {e}")
                    
                    if len(embeddings) > 1:
                        embeddings_array = np.array(embeddings)
                        distances = cosine_distances(embeddings_array)
                        
                        field_diversity[field] = {
                            "semantic_diversity": float(np.mean(distances)),
                            "semantic_variance": float(np.var(distances)),
                            "sample_values": values[:3],
                            "value_count": len(values)
                        }
            except Exception as e:
                print(f"Warning: Error calculating diversity for field {field}: {e}")
        
        # Identify semantically similar fields that might be duplicates
        try:
            similar_field_groups = self._identify_similar_fields(field_embeddings)
        except Exception as e:
            print(f"Warning: Error identifying similar fields: {e}")
            similar_field_groups = []
        
        # Combine all analyses
        analysis = {
            "cross_response_key_consistency": key_consistency,
            "structural_similarity": structural_similarity,
            "type_consistency": type_consistency,
            "field_specific_diversity": field_diversity,
            "similar_field_groups": similar_field_groups
        }
        
        return analysis
    
    def _identify_embedding_clusters(self, similarity_matrix: np.ndarray, outputs: List[dict]) -> List[Dict[str, Any]]:
        """Identify clusters using embedding-based similarity."""
        from sklearn.cluster import AgglomerativeClustering
        
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        
        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.25,  # 0.75 similarity threshold
            metric='precomputed',
            linkage='average'
        )
        
        clusters = clustering.fit_predict(distance_matrix)
        
        # Analyze clusters
        cluster_info = []
        unique_clusters = np.unique(clusters)
        
        for cluster_id in unique_clusters:
            cluster_indices = np.where(clusters == cluster_id)[0]
            
            # Calculate intra-cluster similarity
            intra_similarities = []
            for i in cluster_indices:
                for j in cluster_indices:
                    if i < j:
                        intra_similarities.append(similarity_matrix[i, j])
            
            # Find representative (most central member)
            centrality_scores = []
            for idx in cluster_indices:
                avg_sim = np.mean([similarity_matrix[idx, j] for j in cluster_indices if idx != j])
                centrality_scores.append((idx, avg_sim))
            
            representative_idx = max(centrality_scores, key=lambda x: x[1])[0]
            
            cluster_info.append({
                "cluster_id": int(cluster_id),
                "size": len(cluster_indices),
                "member_indices": cluster_indices.tolist(),
                "avg_intra_similarity": float(np.mean(intra_similarities)) if intra_similarities else 1.0,
                "min_intra_similarity": float(np.min(intra_similarities)) if intra_similarities else 1.0,
                "representative_index": int(representative_idx)
            })
        
        return cluster_info
    
    def _calculate_embedding_metrics(self, 
                                   similarity_matrices: Dict[str, np.ndarray],
                                   outputs: List[dict],
                                   embedding_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics specific to embedding-based analysis."""
        semantic_matrix = similarity_matrices['semantic']
        upper_triangle = semantic_matrix[np.triu_indices_from(semantic_matrix, k=1)]
        
        metrics = {
            # Basic semantic similarity metrics
            "semantic_mean": float(np.mean(upper_triangle)),
            "semantic_median": float(np.median(upper_triangle)),
            "semantic_std": float(np.std(upper_triangle)),
            "semantic_min": float(np.min(upper_triangle)),
            "semantic_max": float(np.max(upper_triangle)),
            
            # Embedding-based consistency score
            "embedding_consistency_score": float(
                np.mean(upper_triangle) * (1 - np.std(upper_triangle))
            ),
        }
        
        # Compare with other analyzers if available
        if 'strict' in similarity_matrices:
            strict_upper = similarity_matrices['strict'][np.triu_indices_from(similarity_matrices['strict'], k=1)]
            metrics["strict_mean"] = float(np.mean(strict_upper))
            metrics["embedding_gain"] = metrics["semantic_mean"] - metrics["strict_mean"]
            
            # Calculate gain percentage
            if metrics["strict_mean"] > 0:
                metrics["embedding_gain_percentage"] = (metrics["embedding_gain"] / metrics["strict_mean"]) * 100
            else:
                metrics["embedding_gain_percentage"] = 100.0
        
        if 'structure_focused' in similarity_matrices:
            struct_upper = similarity_matrices['structure_focused'][np.triu_indices_from(similarity_matrices['structure_focused'], k=1)]
            metrics["structure_consistency"] = float(np.mean(struct_upper))
        
        if 'value_focused' in similarity_matrices:
            value_upper = similarity_matrices['value_focused'][np.triu_indices_from(similarity_matrices['value_focused'], k=1)]
            metrics["value_consistency"] = float(np.mean(value_upper))
        
        if 'high_threshold' in similarity_matrices:
            high_upper = similarity_matrices['high_threshold'][np.triu_indices_from(similarity_matrices['high_threshold'], k=1)]
            metrics["high_threshold_consistency"] = float(np.mean(high_upper))
        
        # Add embedding diversity metrics if available
        if embedding_analysis:
            metrics["key_embedding_diversity"] = embedding_analysis.get("embedding_diversity", 0.0)
            metrics["value_embedding_diversity"] = embedding_analysis.get("value_embedding_diversity", 0.0)
            metrics["num_embedding_clusters"] = len(embedding_analysis.get("embedding_clusters", []))
        
        return metrics
    
    def _perform_detailed_embedding_analysis(self, outputs: List[dict]) -> List[Dict[str, Any]]:
        """Perform detailed pairwise analysis using embeddings."""
        detailed_comparisons = []
        
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                # Get detailed comparison from semantic analyzer
                comparison = self.detailed_comparison(
                    outputs[i], outputs[j]
                )
                
                # Extract key mapping information
                key_mappings = []
                if 'key_analysis' in comparison['details']:
                    analysis = comparison['details']['key_analysis']
                    
                    # Get semantic matches
                    for match in analysis.get('semantic_matches', []):
                        key_mappings.append({
                            'from': match['key1'],
                            'to': match['key2'],
                            'similarity': match['similarity'],
                            'type': 'semantic'
                        })
                    
                    # Get exact matches
                    for key in analysis.get('exact_matches', []):
                        key_mappings.append({
                            'from': key,
                            'to': key,
                            'similarity': 1.0,
                            'type': 'exact'
                        })
                
                detailed_comparisons.append({
                    "output_indices": [i, j],
                    "overall_similarity": comparison["overall_similarity"],
                    "key_mappings": key_mappings,
                    "unmapped_keys": {
                        "output1": comparison['details'].get('key_analysis', {}).get('keys1_only', []),
                        "output2": comparison['details'].get('key_analysis', {}).get('keys2_only', [])
                    }
                })
        
        return detailed_comparisons
    
    def _flatten_keys(self, obj: dict, prefix: str = "", include_nested: bool = True) -> set:
        """Flatten nested dictionary keys."""
        keys = set()
        
        def flatten(o, p):
            if isinstance(o, dict):
                for k, v in o.items():
                    new_key = f"{p}.{k}" if p else k
                    keys.add(new_key)
                    
                    if include_nested:
                        flatten(v, new_key)
            elif include_nested and isinstance(o, list) and o and isinstance(o[0], dict):
                # Add array notation
                keys.add(f"{p}[]")
                # Also analyze first element structure
                flatten(o[0], f"{p}[0]")
        
        flatten(obj, prefix)
        return keys
    
    def _extract_string_values(self, obj: Any) -> List[str]:
        """Extract all string values from nested structure."""
        strings = []
        
        def extract(o):
            if isinstance(o, str):
                strings.append(o)
            elif isinstance(o, dict):
                for v in o.values():
                    extract(v)
            elif isinstance(o, list):
                for item in o:
                    extract(item)
        
        extract(obj)
        return strings
        
    def _extract_field_values(self, obj: Any, prefix: str = "") -> List[Tuple[str, Any]]:
        """Extract all field paths and their values from an object."""
        field_values = []
        
        def extract(o, p):
            if isinstance(o, dict):
                for k, v in o.items():
                    current_path = f"{p}.{k}" if p else k
                    field_values.append((current_path, v))
                    extract(v, current_path)
            elif isinstance(o, list) and o and isinstance(o[0], dict):
                # Handle arrays of objects - extract first item as representative
                extract(o[0], f"{p}[0]")
        
        extract(obj, prefix)
        return field_values
        
    def _identify_similar_fields(self, field_embeddings: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Identify groups of semantically similar fields that might be duplicates."""
        if not field_embeddings or not self.embedding_model:
            return []
        
        # Get field names and representative values
        fields = list(field_embeddings.keys())
        if len(fields) < 2:
            return []
        
        # Use parent class's key mapping functionality
        key_mapping = self._find_key_mapping(fields, fields)
        
        # Find groups of similar fields
        similar_groups = []
        processed = set()
        
        for field1 in fields:
            if field1 in processed:
                continue
                
            # Find fields similar to this one
            similar_fields = []
            for field2 in fields:
                if field1 != field2 and field2 in key_mapping.get(field1, {}):
                    similarity = self._calculate_key_similarity(field1, field2)
                    if similarity > 0.85:  # High similarity threshold
                        similar_fields.append({
                            "field": field2,
                            "similarity": float(similarity)
                        })
            
            if similar_fields:
                similar_groups.append({
                    "base_field": field1,
                    "similar_fields": similar_fields,
                    "sample_values": field_embeddings[field1][:2]
                })
                processed.add(field1)
                for item in similar_fields:
                    processed.add(item["field"])
        
        return similar_groups
    
    def run_consistency_analysis(self, outputs: List[Union[str, dict]], use_async: bool = False) -> Dict[str, Any]:
        """Run consistency analysis on outputs with option for async processing.
        
        Args:
            outputs: List of JSON outputs to analyze
            use_async: Whether to use async processing for better performance
            
        Returns:
            Dictionary with consistency analysis results
        """
        try:
            if not outputs:
                return {
                    'timestamp': datetime.now().isoformat(),
                    'error': "No outputs provided for analysis",
                    'num_outputs_analyzed': 0
                }
                
            if use_async and EMBEDDINGS_AVAILABLE:
                # Run async version if supported
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    return loop.run_until_complete(self.get_field_consistency_report_async(outputs))
                except Exception as e:
                    print(f"Error in async processing: {e}. Falling back to synchronous version.")
                    return self.get_field_consistency_report(outputs)
            else:
                # Use synchronous version
                return self.get_field_consistency_report(outputs)
        except Exception as e:
            print(f"Critical error in consistency analysis: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': f"Critical error in consistency analysis: {str(e)}",
                'num_outputs_analyzed': len(outputs) if outputs else 0
            }

    def print_field_consistency_summary(self, report: Dict[str, Any]):
        """Print a readable summary of field consistency report."""
        if report is None:
            print("Error: No report available to summarize.")
            return
            
        if "error" in report:
            print(f"Error in field consistency report: {report['error']}")
            return
            
        print("=== Field-Level Consistency Report ===\n")
        
        try:
            print(f"Analyzed {report.get('num_outputs_analyzed', 0)} outputs")
            print(f"Found {report.get('num_unique_field_groups', 0)} unique field groups\n")
            
            print("📊 Aggregate Metrics:")
            agg = report.get('aggregate_metrics', {})
            print(f"  • Mean field consistency: {agg.get('mean_field_consistency', 0.0):.3f}")
            print(f"  • Std deviation: {agg.get('std_field_consistency', 0.0):.3f}")
            print(f"  • Range: [{agg.get('min_field_consistency', 0.0):.3f}, {agg.get('max_field_consistency', 1.0):.3f}]")
            
            print(f"\n🏆 Most Consistent Fields:")
            for field_info in report.get('most_consistent_fields', [])[:3]:
                print(f"  • {field_info.get('field', 'Unknown')}: {field_info.get('consistency_score', 0.0):.3f}")
                if 'key_variations' in field_info and len(field_info['key_variations']) > 1:
                    print(f"    Variations: {', '.join(field_info['key_variations'][:3])}")
            
            print(f"\n⚠️  Least Consistent Fields:")
            for field_info in report.get('least_consistent_fields', [])[:3]:
                print(f"  • {field_info.get('field', 'Unknown')}: {field_info.get('consistency_score', 0.0):.3f}")
                field_name = field_info.get('field', '')
                metrics = report.get('field_consistency_scores', {}).get(field_name, {})
                if 'consistency_details' in metrics and isinstance(metrics['consistency_details'], dict):
                    details = metrics['consistency_details']
                    if 'variations' in details:
                        print(f"    Found {len(details['variations'])} different values")
            
            print(f"\n💡 Recommendations:")
            for rec in report.get('recommendations', [])[:5]:
                print(f"{rec}")
        except Exception as e:
            print(f"Error while printing report summary: {e}")
            print("Report structure may be incomplete or invalid.")
            print("Available keys in report:", list(report.keys()) if isinstance(report, dict) else "None")


# Example usage with SemanticJSONSimilarity
if __name__ == "__main__":
    # Simulate outputs with semantic variations
    outputs = [
        {
            "user_name": "John Doe",
            "user_age": 30,
            "email_address": "john@example.com",
            "home_address": {
                "street_name": "123 Main St",
                "city_name": "New York",
                "postal_code": "10001"
            },
            "skills": ["Python", "JavaScript", "SQL"],
            "years_of_experience": 5
        },
        {
            "name": "John Doe",  # Semantic variation
            "age": 30,
            "email": "john@example.com",
            "address": {  # Semantic variation
                "street": "123 Main St",
                "city": "New York",
                "zip": "10001"  # Semantic variation
            },
            "skills": ["Python", "JS", "SQL"],  # JS vs JavaScript
            "experience_years": 5  # Semantic variation
        },
        {
            "userName": "John Doe",  # camelCase variation
            "userAge": 30,
            "emailAddr": "john@example.com",
            "homeAddr": {
                "streetName": "123 Main St",
                "cityName": "New York",
                "zipCode": "10001"
            },
            "skillset": ["Python", "JavaScript", "SQL"],
            "exp_years": 5
        },
        {
            "full_name": "John Doe",  # Different semantic variation
            "age": 30,
            "contact_email": "john@example.com",
            "residential_address": {
                "street_address": "123 Main St",
                "city": "New York",
                "zip_code": "10001"
            },
            "technical_skills": ["Python", "JavaScript", "SQL", "React"],
            "experience": 5
        }
    ]
    
    # Create embedding-based analyzer
    analyzer = EmbeddingBasedConsistencyAnalyzer(
        semantic_similarity_config={
            'structure_weight': 0.4,
            'value_weight': 0.6,
            'semantic_threshold': 0.7,
            'use_semantic_similarity': True,
            'string_method': 'semantic'  # Use embeddings for string comparison too
        },
        embedding_model='all-MiniLM-L6-v2'  # You can change this to other models
    )
    
    # Analyze consistency
    print("Analyzing consistency with embedding-based semantic understanding...\n")
    
    result = analyzer.analyze_embedding_consistency(
        outputs=outputs,
        prompt="Extract person information including name, age, email, address, skills, and experience",
        model_params={"temperature": 0.7, "model": "gpt-3.5-turbo"},
        detailed_analysis=True,
        analyze_embeddings=True
    )
    
    # Show specific insights
    print("\n\n=== Embedding-specific Insights ===")
    print(f"\nEmbedding model used: {result.get('embedding_model_used', 'None')}")
    print(f"Embedding gain over strict matching: {result['metrics'].get('embedding_gain', 0):.3f}")
    print(f"Number of semantic clusters: {len(result.get('semantic_clusters', []))}")
    
    key_analysis = result.get('key_embedding_analysis', {})
    if key_analysis.get('semantic_groups'):
        print(f"\nFound {len(key_analysis['semantic_groups'])} semantic key groups:")
        for group in key_analysis['semantic_groups'][:3]:
            print(f"  - {', '.join(group['keys'])}")
            
            
    # Get field consistency report
    field_report = analyzer.get_field_consistency_report(
        outputs=outputs,
        include_nested=True,
        min_presence_ratio=0.5
    )

    # Print summary
    analyzer.print_field_consistency_summary(field_report)