from typing import Dict, Any, List, Tuple, Union
import json
import numpy as np
import time
import warnings
import boto3
from botocore.config import Config

def collect_all_values(data: Union[Dict, List, Any]) -> List[Tuple[str, Any]]:
    """
    Collect all values from a nested dictionary/list structure.
    
    Args:
        data: The input data structure (dict, list, or primitive)
        include_keys: If True, also collect dictionary keys as values
        
    Returns:
        List of tuples (path, value) where path shows the location of the value
    """
    values = []
    def _collect_recursive(obj):
        """Recursively collect values with their paths."""
        if isinstance(obj, dict):
            # Collect dictionary values
            for key, value in obj.items():
                _collect_recursive(value)

        elif isinstance(obj, list):
            # Collect list elements
            for idx, item in enumerate(obj):
                _collect_recursive(item)

        else:
            # Leaf value (string, number, boolean, None, etc.)
            if isinstance(obj, str):
                values.append(obj)

    _collect_recursive(data)
    return values

def count_json_elements(json_obj: Any) -> int:
    """
    Count the total number of elements in a JSON structure.
    Used for normalizing structural penalties.
    
    Args:
        json_obj: JSON object to count elements in
        
    Returns:
        Total number of elements (keys, values, array items)
    """
    if isinstance(json_obj, dict):
        return len(json_obj) + sum(count_json_elements(v) for v in json_obj.values())
    elif isinstance(json_obj, list):
        return len(json_obj) + sum(count_json_elements(item) for item in json_obj)
    else:
        return 1  # Primitive value counts as 1 element

def parse_json_outputs(outputs: List[Union[str, Dict]]) -> List[Dict]:
    """
    Parse a list of JSON outputs that might be strings or dictionaries.
    
    Args:
        outputs: List of JSON strings or dictionaries
        
    Returns:
        List of parsed JSON dictionaries
    """
    parsed = []
    for i, output in enumerate(outputs):
        if isinstance(output, str):
            try:
                parsed.append(json.loads(output))
            except json.JSONDecodeError as e:
                warnings.warn(f"Could not parse JSON string at index {i}: {e}")
        elif isinstance(output, list):
            if isinstance(output[0], dict) and len(output)>0:
                parsed.append({"responses": output})
            elif isinstance(output[0], str) and len(output)>0:
                parsed.append({"responses": json.loads(str)})
            else:
                parsed.append({"responses": output})
        elif isinstance(output, dict):
            parsed.append(output)
        else:
            parsed.append({"responses": output})

    return parsed

def get_embeddings(text, model_id, bedrock_client, max_retries=10, initial_delay=2, output_embedding_length=1024):
    """
    Get embeddings from Bedrock with proper retry and connection handling
    
    Args:
        text: Input text to embed
        model_id: Bedrock model ID
        bedrock_client: Boto3 Bedrock client
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        output_embedding_length: Expected embedding dimension
        
    Returns:
        numpy array of embeddings
    """
    # Validate input text
    if not text or not text.strip():
        warnings.warn(f"Empty or whitespace-only text provided for embedding: '{text}'")
        return np.zeros(output_embedding_length)

    def exponential_delay(attempt):
        # Add jitter to prevent thundering herd
        jitter = 0.1 * initial_delay * np.random.random()
        return initial_delay * (2 ** attempt) + jitter

    if model_id == "amazon.titan-embed-text-v1":
        request_body = {
            "inputText": text
        }
    elif model_id == "amazon.titan-embed-text-v2:0":
        request_body = {
            "inputText": text,
            "dimensions": output_embedding_length,
            "normalize": True,
            "embeddingTypes": ["float"]
        }
    elif model_id == "cohere.embed-multilingual-v3":
        request_body = {
            "texts": [text],
            "input_type": "clustering",
            "truncate": "END",
            "dimensions": output_embedding_length,
            "normalize": True,
            "embeddingTypes": ["float"]
        }
    else:
        raise ValueError(f"Unknown model_id: {model_id}")

    for attempt in range(max_retries):
        try:
            body = json.dumps(request_body)
            response = bedrock_client.invoke_model(
                body=body,
                modelId=model_id,
                accept="application/json",
                contentType="application/json")
            response_body = json.loads(response.get("body").read())

            # Handle different response formats
            if model_id == "cohere.embed-multilingual-v3":
                embedding = response_body.get("embeddings", [[]])[0]
            else:
                embedding = response_body.get("embedding", [])

            return np.array(embedding).astype(np.float32)
        except Exception:
            if attempt == max_retries - 1:
                raise

            delay = exponential_delay(attempt)
            time.sleep(delay)

    # This should be unreachable with the retry logic above
    raise Exception("Max retries reached. Unable to get embeddings.")


def create_bedrock_client(region_name="us-west-2", config=None):
    """
    Create a configured Bedrock client with optimized settings.
    
    Args:
        region_name: AWS region name
        
    Returns:
        Configured boto3 Bedrock client
    """
    if config is None:
        config = Config(
            max_pool_connections=50,  # Increase connection pool size
            retries={'max_attempts': 5, 'mode': 'adaptive'},
            connect_timeout=10,
            read_timeout=30,
            tcp_keepalive=True
        )
    session = boto3.Session()

    return session.client(
        'bedrock-runtime',
        config=config,
        region_name=region_name
    )
