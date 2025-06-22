import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
import json
from collections import defaultdict
import asyncio
import concurrent.futures
from functools import partial
import aiohttp
from tqdm import tqdm

class FieldAwareConsistencyCalculator:
    def __init__(
        self,
        bedrock_client,
        eval_fields, 
        result_field_name, 
        embedding_model_id="amazon.titan-embed-text-v2:0",
        embed_dimension=256,
        similarity_threshold=0.7, 
        empty_set_handling="zero", 
        max_concurrent_embeddings=10,
        primary_field=None,
        min_value_threshold=0.6,
        nested_field_separator="."  # Add this parameter
    ):
        """
        Initialize the calculator that measures consistency separately by field
        
        Parameters:
        -----------
        bedrock_client : obj
            Bedrock client for embedding API calls
        embedding_model_id : str
            ID of the embedding model to use
        embed_dimension : int
            Dimension of embeddings to generate
        similarity_threshold : float
            Threshold above which two elements are considered semantically similar
        empty_set_handling : str
            How to handle empty sets:
            - "zero": empty sets have zero similarity with any other set
            - "one": empty sets have similarity 1.0 with other empty sets, 0.0 otherwise
            - "partial": empty sets similarity is proportional to set size difference
        max_concurrent_embeddings : int
            Maximum number of concurrent embedding API calls
        """
        self.similarity_threshold = similarity_threshold
        self.empty_set_handling = empty_set_handling
        self.bedrock_client = bedrock_client
        self.embed_dimension = embed_dimension
        self.embedding_model_id = embedding_model_id
        self.max_concurrent_embeddings = max_concurrent_embeddings
        self.embedding_cache = {}  # Cache for embeddings to avoid recomputing
        self.result_field_name = result_field_name
        self.primary_field = primary_field
        self.min_value_threshold = min_value_threshold
        
        # Fields to extract and analyze separately
        self.fields = eval_fields
        
        self.nested_field_separator = nested_field_separator
        
        # Parse fields to identify nested paths
        self.fields = eval_fields
        self.field_paths = {}
        for field in eval_fields:
            if self.nested_field_separator in field:
                self.field_paths[field] = field.split(self.nested_field_separator)
            else:
                self.field_paths[field] = [field]
        
    def find_most_common_values(self, responses, field):
        """
        Find the most common values for a specific field across responses
        
        Parameters:
        -----------
        responses : list
            List of response dictionaries
        field : str
            Field name to analyze
                
        Returns:
        --------
        list
            Most common values for the field, with their frequencies
        """
        # Extract field values from each response
        all_values = []
        for response in responses:
            field_values = self.extract_fields(response)
            all_values.extend(field_values[field])
        
        # If there are no values, return empty result
        if not all_values:
            return []
        
        # For string values, we need to handle semantic similarity
        if all(isinstance(val, str) for val in all_values):
            # Group semantically similar values
            clusters = []
            
            for value in all_values:
                # Check if this value belongs to an existing cluster
                assigned = False
                for cluster in clusters:
                    # Compare with the representative of the cluster
                    similarity = self.calculate_element_similarity(value, cluster['representative'])
                    if similarity >= self.similarity_threshold:
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
                'variations': cluster['values']
            } for cluster in sorted_clusters]
        
        # For non-string values, use direct counting
        else:
            from collections import Counter
            value_counts = Counter(all_values)
            
            # Return most common values with their frequencies
            return [{
                'value': value,
                'count': count,
                'variations': [value]
            } for value, count in value_counts.most_common()]
    
    def _set_nested_value(self, dictionary, key_path, value):
        """
        Set value in nested dictionary using dot notation
        
        Parameters:
        -----------
        dictionary : dict
            The dictionary to modify
        key_path : str
            The path where to set the value (e.g., "user.address.city")
        value : any
            The value to set
        """
        if '.' in key_path:
            keys = key_path.split('.')
            current = dictionary
            
            # Navigate to the parent of the final key
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set the final value
            current[keys[-1]] = value
        else:
            dictionary[key_path] = value
            
    def analyze_common_values_and_corresponding_fields(self, responses):
        """
        Identify most common values for a primary field, then analyze consistency
        of corresponding fields for each common primary value
        
        Parameters:
        -----------
        responses : list
            List of response dictionaries
                
        Returns:
        --------
        list
            Analysis of most common primary values and their corresponding field consistency
        """
        # First, flatten all the corrections from all responses
        all_corrections = []
        for response in responses:
            # Get nested data
            corrections = self._get_nested_value(response, self.result_field_name)
            
            if not corrections:
                continue
                
            # Ensure it's a list
            if not isinstance(corrections, list):
                corrections = [corrections]
                
            for correction in corrections:
                primary_value = self._get_nested_value(correction, self.primary_field)
                if primary_value:
                    all_corrections.append(correction)
        
        # If no corrections found, return empty result
        if not all_corrections:
            return []
        
        # Group corrections by the primary field using semantic similarity
        primary_clusters = []
        
        for correction in all_corrections:
            primary_value = correction[self.primary_field]
            
            # Skip if the primary value is missing
            if not primary_value:
                continue
            
            # Check if this value belongs to an existing cluster
            assigned = False
            for cluster in primary_clusters:
                # Compare with the representative of the cluster
                similarity = self.calculate_element_similarity(primary_value, cluster['representative'])
                if similarity >= self.similarity_threshold:
                    cluster['corrections'].append(correction)
                    cluster['values'].append(primary_value)
                    cluster['count'] += 1
                    assigned = True
                    break
            
            # If not assigned to any existing cluster, create a new one
            if not assigned:
                primary_clusters.append({
                    'representative': primary_value,
                    'values': [primary_value],
                    'corrections': [correction],
                    'count': 1
                })
        
        # Sort clusters by count in descending order
        sorted_clusters = sorted(primary_clusters, key=lambda x: x['count'], reverse=True)
        
        # For each common primary value, analyze the corresponding fields
        results = []
        
        for cluster in sorted_clusters:
            # For each corresponding field, analyze consistency
            percentage = cluster['count'] / len(responses) if responses else 0
            if percentage < self.min_value_threshold:
                continue
            
            # Add result for this primary value cluster
            results.append({
                "representative": cluster['representative'],
                "record": cluster['corrections'],
                "count": cluster['count']
                })
        
        return results
    
    def analyze_corrections_consistency(self, data_list):
        """
        Analyze consistency of corrections across multiple responses
        
        Parameters:
        -----------
        data_list : list
            List of response dictionaries with 'corrections' field
                
        Returns:
        --------
        dict
            Analysis results with common primary values and their corresponding field consistency
        """
        # Reformat data if needed to match expected structure
        formatted_data = []
        for item in data_list:
            formatted_data.append({self.result_field_name: item.get(self.result_field_name, [])})
        
        # Use the main analysis function
        analysis = self.analyze_common_values_and_corresponding_fields(formatted_data)
        
        # Calculate overall statistics
        total_primary_values = sum(cluster['count'] for cluster in analysis)
        
        # Format results for better readability
        result = {
            "total_corrections": total_primary_values,
            "common_primary_values": len(analysis),
            "analysis": analysis
        }
        
        return result
    
    def embed_text(self, input_text: str) -> np.ndarray:
        """Synchronous embedding (for backward compatibility)"""
        # Check cache first
        if input_text in self.embedding_cache:
            return self.embedding_cache[input_text]
        
        # Create the request for the model.
        native_request = {
            "inputText": input_text,
            "dimensions": self.embed_dimension
        }
        
        # Convert the native request to JSON.
        request = json.dumps(native_request)

        # Invoke the model with the request.
        response = self.bedrock_client.invoke_model(
            modelId=self.embedding_model_id,
            body=request,
            accept="application/json",
            contentType="application/json"
        )

        # Decode the model's native response body.
        model_response = json.loads(response["body"].read())

        # Extract the embedding
        embedding = model_response["embedding"]
        
        # Cache the result
        self.embedding_cache[input_text] = embedding
        return embedding
    
    async def embed_text_async(self, input_text: str) -> np.ndarray:
        """Asynchronous embedding function"""
        # Check cache first
        if input_text in self.embedding_cache:
            return self.embedding_cache[input_text]
        
        # Create the request for the model
        native_request = {
            "inputText": input_text,
            "dimensions": self.embed_dimension
        }
        
        # Convert the native request to JSON
        request = json.dumps(native_request)
        
        # Use a thread pool for the blocking bedrock client call
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            response = await loop.run_in_executor(
                executor,
                partial(
                    self.bedrock_client.invoke_model,
                    modelId=self.embedding_model_id,
                    body=request,
                    accept="application/json",
                    contentType="application/json"
                )
            )
        
        # Decode the model's native response body
        body_bytes = await loop.run_in_executor(executor, response["body"].read)
        model_response = json.loads(body_bytes)
        
        # Extract the embedding
        embedding = model_response["embedding"]
        
        # Cache the result
        self.embedding_cache[input_text] = embedding
        return embedding
    
    async def batch_embed_texts(self, texts):
        """Embed multiple texts concurrently with rate limiting"""
        # Create a semaphore to limit concurrent API calls
        semaphore = asyncio.Semaphore(self.max_concurrent_embeddings)
        
        async def embed_with_semaphore(text):
            async with semaphore:
                return await self.embed_text_async(text)
        
        # Process all texts concurrently
        tasks = [embed_with_semaphore(text) for text in texts]
        return await asyncio.gather(*tasks)
    
    def extract_fields(self, response):
        """
        Extract fields from a response object, handling nested dictionaries
        
        Parameters:
        -----------
        response : dict
            Response dictionary that may contain nested structures
            
        Returns:
        --------
        dict
            Dictionary mapping field names to lists of field values
        """
        field_values = {field: [] for field in self.fields}
        
        if not response:
            return field_values
        
        # Handle case where result_field_name might be nested
        data = self._get_nested_value(response, self.result_field_name)
        
        if not data:
            return field_values
            
        # Ensure data is a list
        if not isinstance(data, list):
            data = [data]
        
        for item in data:
            for field in self.fields:
                value = self._get_nested_value(item, field)
                if value is not None:
                    field_values[field].append(value)
        
        return field_values
    
    def _get_nested_value(self, dictionary, key_path):
        """
        Get value from nested dictionary using dot notation
        
        Parameters:
        -----------
        dictionary : dict
            The dictionary to search in
        key_path : str
            The path to the value (e.g., "user.address.city")
            
        Returns:
        --------
        any
            The value at the specified path, or None if not found
        """
        if not dictionary:
            return None
            
        # If key_path contains dots, it's a nested path
        if '.' in key_path:
            keys = key_path.split('.')
            value = dictionary
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                elif isinstance(value, list) and key.isdigit():
                    # Handle list indices
                    try:
                        value = value[int(key)]
                    except (IndexError, ValueError):
                        return None
                else:
                    return None
                    
            return value
        else:
            # Simple key lookup
            return dictionary.get(key_path)
    
    async def calculate_element_similarity_async(self, elem1, elem2):
        """Calculate semantic similarity between two elements asynchronously"""
        # For text elements, use embeddings and cosine similarity
        if isinstance(elem1, str) and isinstance(elem2, str):
            embedding1 = await self.embed_text_async(elem1)
            embedding2 = await self.embed_text_async(elem2)
            
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            
            # Return the actual similarity score
            return float(similarity)
        
        # For exact matches of non-string types
        elif elem1 == elem2:
            return 1.0
        else:
            return 0.0
    
    def calculate_element_similarity(self, elem1, elem2):
        """Synchronous version of similarity calculation"""
        # For text elements, use embeddings and cosine similarity
        if isinstance(elem1, str) and isinstance(elem2, str):
            embedding1 = self.embed_text(elem1)
            embedding2 = self.embed_text(elem2)
            
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            
            # Return the actual similarity score
            return float(similarity)
        
        # For exact matches of non-string types
        elif elem1 == elem2:
            return 1.0
        else:
            return 0.0
    
    async def calculate_set_similarity_async(self, set1, set2, return_pairs=False, filter_threshold=0.5):
        """
        Calculate similarity between two sets of elements using maximum bipartite matching
        
        Parameters:
        -----------
        set1, set2 : list
            Lists of string elements
            
        Returns:
        --------
        float
            Similarity score between 0 and 1
        """
        # Handle empty sets
        if not set1 and not set2:
            return 1.0  # Two empty sets are identical
        
        if not set1 or not set2:
            if self.empty_set_handling == "zero":
                return 0.0
            elif self.empty_set_handling == "one":
                return 0.0  # Empty set vs non-empty is always 0
            elif self.empty_set_handling == "partial":
                # Return a score based on the relative size difference
                max_size = max(len(set1), len(set2))
                return 1.0 / (1 + max_size)  # Approaches 0 as set size increases
        
        # Precompute all embeddings in parallel to avoid redundant API calls
        all_texts = list(set([elem for elem in set1 + set2 if isinstance(elem, str)]))
        if all_texts:
            await self.batch_embed_texts(all_texts)
        
        # Create similarity matrix
        similarity_matrix = np.zeros((len(set1), len(set2)))
        
        # Calculate similarities in parallel
        tasks = []
        for i, elem1 in enumerate(set1):
            for j, elem2 in enumerate(set2):
                tasks.append((i, j, self.calculate_element_similarity_async(elem1, elem2)))
        
        # Await all similarity calculations
        for i, j, task in tasks:
            similarity_matrix[i, j] = await task
        
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
    
    def calculate_set_similarity(self, set1, set2, return_pairs=False, filter_threshold=0.5):
        """Synchronous version of set similarity calculation"""
        # Handle empty sets
        if not set1 and not set2:
            return 1.0  # Two empty sets are identical
        
        if not set1 or not set2:
            if self.empty_set_handling == "zero":
                return 0.0
            elif self.empty_set_handling == "one":
                return 0.0  # Empty set vs non-empty is always 0
            elif self.empty_set_handling == "partial":
                # Return a score based on the relative size difference
                max_size = max(len(set1), len(set2))
                return 1.0 / (1 + max_size)  # Approaches 0 as set size increases
        
        # Create similarity matrix
        similarity_matrix = np.zeros((len(set1), len(set2)))
        for i, elem1 in enumerate(set1):
            for j, elem2 in enumerate(set2):
                similarity_matrix[i, j] = self.calculate_element_similarity(elem1, elem2)
        
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
    
    async def calculate_field_consistency_async(self, responses, field):
        """
        Calculate consistency for a specific field across responses
        
        Parameters:
        -----------
        responses : list
            List of response dictionaries
        field : str
            Field name to analyze
            
        Returns:
        --------
        float
            Consistency score for the field
        dict
            Additional metrics for the field
        """
        # Extract field values from each response
        field_sets = []
        for response in responses:
            field_values = self.extract_fields(response)
            field_sets.append(field_values[field])
        
        n = len(field_sets)
        if n <= 1:
            return 1.0, {"pairs": 0, "empty_count": 1 if n == 1 and not field_sets[0] else 0}
        
        # Count empty sets
        empty_count = sum(1 for field_set in field_sets if not field_set)
        
        # Calculate pairwise similarities concurrently
        similarities = []
        tasks = []
        
        for i in range(n-1):
            for j in range(i+1, n):
                tasks.append((i, j, self.calculate_set_similarity_async(field_sets[i], field_sets[j])))
        
        # Await all similarity calculations
        for i, j, task in tasks:
            sim = await task
            similarities.append((i, j, sim))
        
        # Calculate total similarity
        total_similarity = sum(sim for _, _, sim in similarities)
        pair_count = len(similarities)
        
        # Average pairwise similarity
        consistency_score = total_similarity / pair_count if pair_count > 0 else 1.0
        
        # Find most common values
        # most_common = self.find_most_common_values(responses, field)
        consistency_analysis = ""
        if field == self.primary_field:
            consistency_analysis = self.analyze_corrections_consistency(responses)
        
        # Additional metrics
        metrics = {
            #"pairs": pair_count,
            "empty_count": empty_count,
            "empty_percentage": empty_count / n if n > 0 else 0,
            #"pairwise_similarities": similarities,
            #"element_counts": [len(fs) for fs in field_sets],
            "consistency_analysis": consistency_analysis
        }
        
        return consistency_score, metrics
    
    def calculate_field_consistency(self, responses, field):
        """Synchronous version of field consistency calculation"""
        # Extract field values from each response
        field_sets = []
        
        for response in responses:
            field_values = self.extract_fields(response)
            field_sets.append(field_values[field])
        
        n = len(field_sets)
        if n <= 1:
            return 1.0, {"pairs": 0, "empty_count": 1 if n == 1 and not field_sets[0] else 0}
        
        # Count empty sets
        empty_count = sum(1 for field_set in field_sets if not field_set)
        
        # Calculate pairwise similarities
        total_similarity = 0.0
        pair_count = 0
        similarities = []
        
        for i in range(n-1):
            for j in range(i+1, n):
                sim = self.calculate_set_similarity(field_sets[i], field_sets[j])
                similarities.append((i, j, sim))
                total_similarity += sim
                pair_count += 1
        
        # Average pairwise similarity
        consistency_score = total_similarity / pair_count if pair_count > 0 else 1.0
        
        # Find most common values
        # most_common = self.find_most_common_values(responses, field)
        consistency_analysis = ""
        if field == self.primary_field:
            consistency_analysis = self.analyze_corrections_consistency(responses)
        
        # Additional metrics
        metrics = {
            #"pairs": pair_count,
            "empty_count": empty_count,
            "empty_percentage": empty_count / n if n > 0 else 0,
            #"pairwise_similarities": similarities,
            #"element_counts": [len(fs) for fs in field_sets],
            "consistency_analysis": consistency_analysis
        }
        
        return consistency_score, metrics
    
    async def calculate_prompt_consistency_async(self, responses):
        """
        Calculate consistency score for multiple runs of the same prompt,
        broken down by field and combined
        
        Parameters:
        -----------
        responses : list
            List of response dictionaries, each containing 'corrections' field
            
        Returns:
        --------
        dict
            Dictionary with consistency scores for each field and combined
        dict
            Additional metrics for each field
        """
        field_consistency = {}
        field_metrics = {}
        
        # Calculate consistency for each field concurrently
        tasks = {field: self.calculate_field_consistency_async(responses, field) 
                 for field in self.fields}
        
        # Await all field consistency calculations
        for field, task in tasks.items():
            score, metrics = await task
            field_consistency[field] = score
            field_metrics[field] = metrics
        
        # Calculate overall consistency (average of field consistencies)
        if field_consistency:
            overall_consistency = sum(field_consistency.values()) / len(field_consistency)
        else:
            overall_consistency = 1.0
            
        
        pred = []
        if field_metrics[self.primary_field]['consistency_analysis']:
            pred = field_metrics[self.primary_field]['consistency_analysis']['analysis']
        
        # Combine results
        results = {
            "consistency_score": {
                "overall": overall_consistency,
                **field_consistency,
            },
            "pred": pred,
        }
        
        return results, field_metrics
    
    def calculate_prompt_consistency(self, responses):
        """Synchronous version of prompt consistency calculation"""
        field_consistency = {}
        field_metrics = {}
        
        # Calculate consistency for each field
        for field in self.fields:
            score, metrics = self.calculate_field_consistency(responses, field)
            field_consistency[field] = score
            field_metrics[field] = metrics
        
        # Calculate overall consistency (average of field consistencies)
        if field_consistency:
            overall_consistency = sum(field_consistency.values()) / len(field_consistency)
        else:
            overall_consistency = 1.0
            
        pred = []
        if field_metrics[self.primary_field]['consistency_analysis']:
            pred = field_metrics[self.primary_field]['consistency_analysis']['analysis']
        
        # Combine results
        results = {
            "consistency_score": {
                "overall": overall_consistency,
                **field_consistency,
            },
            "pred": pred,
        }
        
        return results, field_metrics
    
    def calculate_all_metrics(self, pred_dict, gt_df, eval_fields, metric_save_path=None):
        # Part 1: Calculate consistency between predictions for each file
        file_consistency_scores = {}
        pred_result_list = {}

        for file_id, predictions in pred_dict.items():
            consistency_eval, _ = self.calculate_prompt_consistency(predictions)
            file_consistency_scores[file_id] = consistency_eval['consistency_score']
            pred_result_list[file_id] = consistency_eval['pred']

        # Calculate average consistency across all files
        avg_consistency = {
            field: np.mean([scores[field] for scores in file_consistency_scores.values()])
            for field in eval_fields
        }
        
        # Part 2: Calculate metrics based on predictions and ground truth
        all_metrics = {}
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0
        
        # First, build the universe of all possible items
        universe = set()
        for file_id, preds in pred_result_list.items():
            # Add prediction items to universe
            if len(preds) > 0:
                universe.update([pred['representative'] for pred in preds])
            
            if file_id is None:
                continue
            
            # Add ground truth items to universe
            gt_rows = gt_df[gt_df['document_id'] == file_id]
            if len(gt_rows) > 0:
                universe.update([row[self.primary_field] for ind, row in gt_rows.iterrows()])
        
        universe = list(universe)  # Convert to list
        
        for file_id in tqdm(pred_result_list.keys(),  desc='Calculating metrics'):
            if file_id is None:
                continue
            # Get predictions and ground truth
            preds = pred_result_list[file_id]
            gt_rows = gt_df[gt_df['document_id'] == file_id]
            
            pred_list = [pred['representative'] for pred in preds] if len(preds) > 0 else []
            gt_list = [row[self.primary_field] for ind, row in gt_rows.iterrows()] if len(gt_rows) > 0 else []
            
            matched_pairs = []
            if len(pred_list) and len(gt_list):
                # Get similarity and matched pairs
                similarity_score, matched_pairs = self.calculate_set_similarity(pred_list, gt_list, return_pairs=True)
                
                # Calculate metrics
                tp = len(matched_pairs)
                fp = len(pred_list) - tp
                fn = len(gt_list) - tp
                tn = 0
                
            elif len(pred_list) == 0 and len(gt_list) == 0:
                # Both empty - perfect match
                tp = 0
                fp = 0
                fn = 0
                tn = 1  # All items in universe are correctly not predicted
                
            elif len(pred_list) == 0:
                # Missing all ground truth
                tp = 0
                fp = 0
                fn = len(gt_list)
                tn = 0
                
            else:  # len(gt_list) == 0
                # All predictions are false positives
                tp = 0
                fp = len(pred_list)
                fn = 0
                tn = 0
            
            # Accumulate totals
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
            
            # Calculate per-file metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = np.round((tp + tn) / (tp + fp + fn + tn), 4)
            primary_field_consistency = file_consistency_scores[file_id].get(self.primary_field, -1)
            primary_field_consistency = -primary_field_consistency if accuracy == 0 else primary_field_consistency
                    
            # Store file-specific metrics
            # Include both consistency and standard metrics
            all_metrics[file_id] = {
                'primary_field_consistency': primary_field_consistency, 
                'precision': np.round(precision, 4),
                'recall': np.round(recall, 4),
                'f1': np.round(f1, 4),
                'tp': tp,
                'fp': fp,
                'fn': fn, 
                'tn': tn,
                'accuracy': accuracy,
            }
        
        # Calculate overall metrics
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        overall_accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn) if (total_tp + total_fp + total_fn + total_tn) > 0 else 1.0
        
        confusion_matrix = {
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'tn': total_tn
        }
        
        overall_metrics = {
            'overall_avg_consistency': avg_consistency,
            'avg_primary_field_consistency': np.mean([all_metrics[f_id]['primary_field_consistency'] for f_id in pred_result_list.keys()]),
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'accuracy': overall_accuracy,
            'confusion_matrix': confusion_matrix
        }
        
        if metric_save_path:
            # save all_metrics as csv
            with open(metric_save_path, 'w') as f:
                f.write("case_id, primary_field_consistency, precision, recall, f1, accuracy, tp, fp, fn, tn")
                for key, value in all_metrics.items():
                    f.write(f"\n{key}, {value['primary_field_consistency']}, {value['precision']}, {value['recall']}, {value['f1']}, {value['accuracy']}, {value['tp']}, {value['fp']}, {value['fn']}, {value['tn']}")
        
        return all_metrics, overall_metrics, pred_result_list
    
    async def calculate_mean_consistency_async(self, prompt_responses):
        """
        Calculate mean consistency scores across multiple prompts asynchronously
        
        Parameters:
        -----------
        prompt_responses : dict or list
            Dictionary mapping prompts to lists of response dictionaries, or
            List of lists of response dictionaries
            
        Returns:
        --------
        dict
            Mean consistency scores (overall and by field)
        dict
            Detailed metrics per prompt and field
        """
        if isinstance(prompt_responses, dict):
            prompts = list(prompt_responses.keys())
            responses_list = list(prompt_responses.values())
        else:
            prompts = [f"prompt_{i}" for i in range(len(prompt_responses))]
            responses_list = prompt_responses
        
        # Initialize counters for mean calculation
        total_consistency = defaultdict(float)
        detailed_metrics = {}
        
        # Process all prompts concurrently
        tasks = {prompt: self.calculate_prompt_consistency_async(responses) 
                 for prompt, responses in zip(prompts, responses_list)}
        
        # Await all prompt consistency calculations
        for prompt, task in tasks.items():
            consistency_results, metrics = await task
            
            # Add to totals for each field and overall
            for key, score in consistency_results.items():
                total_consistency[key] += score
            
            detailed_metrics[prompt] = {
                "consistency": consistency_results,
                "metrics": metrics
            }
        
        # Calculate means
        num_prompts = len(responses_list)
        mean_consistency = {
            key: value / num_prompts if num_prompts else 1.0
            for key, value in total_consistency.items()
        }
        
        return mean_consistency, detailed_metrics
    
    def calculate_mean_consistency(self, prompt_responses):
        """Synchronous version of mean consistency calculation"""
        if isinstance(prompt_responses, dict):
            prompts = list(prompt_responses.keys())
            responses_list = list(prompt_responses.values())
        else:
            prompts = [f"prompt_{i}" for i in range(len(prompt_responses))]
            responses_list = prompt_responses
        
        # Initialize counters for mean calculation
        total_consistency = defaultdict(float)
        detailed_metrics = {}
        
        for prompt, responses in zip(prompts, responses_list):
            consistency_results, metrics = self.calculate_prompt_consistency(responses)
            
            # Add to totals for each field and overall
            for key, score in consistency_results.items():
                total_consistency[key] += score
            
            detailed_metrics[prompt] = {
                "consistency": consistency_results,
                "metrics": metrics
            }
        
        # Calculate means
        num_prompts = len(responses_list)
        mean_consistency = {
            key: value / num_prompts if num_prompts else 1.0
            for key, value in total_consistency.items()
        }
        
        return mean_consistency, detailed_metrics
    
    # Main entry point for async usage
    async def run_analysis_async(self, prompt_responses):
        """Run the full analysis pipeline asynchronously"""
        return await self.calculate_mean_consistency_async(prompt_responses)
    
    # Helper method to run async code from synchronous context
    def run_analysis(self, prompt_responses):
        """Run the full analysis pipeline using async under the hood"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.run_analysis_async(prompt_responses))