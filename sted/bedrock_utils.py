import json
import logging
import os
from functools import wraps
import time
import asyncio
import random
from concurrent.futures import ThreadPoolExecutor
import boto3
from botocore.config import Config
from botocore.exceptions import ReadTimeoutError

logger = logging.getLogger(__name__)

# Configurable connection pool size via environment variable
MAX_POOL_CONNECTIONS = int(os.environ.get('BEDROCK_MAX_POOL_CONNECTIONS', '100'))

# Configure boto3 with retry settings and increased timeout for slower models like Claude-Opus-4
boto_config = Config(
    retries={
        'max_attempts': 20,
        'mode': 'adaptive'
    },
    max_pool_connections=MAX_POOL_CONNECTIONS,  # Configurable connection pool size
    read_timeout=600,  # 10 minutes timeout for large model responses (e.g., Mistral-Large-675B)
    connect_timeout=30  # 30 seconds to establish connection
)

sm_client = boto3.client("runtime.sagemaker", region_name="us-west-2")

def retry_with_count(max_attempts=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt == max_attempts - 1:
                        return None
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

async def retry_async_with_backoff(func, max_attempts=10, *args, **kwargs):
    """Retry an async function with exponential backoff."""
    for attempt in range(max_attempts):
        try:
            logger.info(f"API call attempt {attempt + 1}/{max_attempts}")
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt == max_attempts - 1:
                logger.error(f"All {max_attempts} attempts failed: {str(e)}")
                # Return None instead of raising to avoid crashing the program
                return None

            # Exponential backoff with jitter
            wait_time = min(2 ** attempt + (0.1 * random.random()), 60)
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time:.2f}s")

            if "ThrottlingException" in str(e) or "TooManyRequestsException" in str(e) or "ServiceUnavailableException" in str(e) or "Too many connections" in str(e):
                # Longer wait for throttling and rate limiting
                wait_time = min(5 ** attempt, 300)  # Increased max wait time for throttling
                logger.warning(f"Rate limiting detected, waiting {wait_time:.2f}s")

            await asyncio.sleep(wait_time)

@retry_with_count(max_attempts=10, delay=5)
def generate_message(
    bedrock_runtime,
    model_id, messages,
    temperature=0.1,
    top_p=0.9,
    top_k=200,
    max_tokens=5000,
    system_prompt="",
):
    bedrock_format = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p
    }

    if system_prompt:
        bedrock_format["system"] = system_prompt

    response = bedrock_runtime.invoke_model(body=json.dumps(bedrock_format), modelId=model_id)
    response_body = json.loads(response.get('body').read())

    return response_body

# create a async generate_message function
@retry_with_count(max_attempts=10, delay=5)
async def async_generate_message(
    bedrock_runtime,
    model_id, messages,
    temperature=0.1,
    top_p=0.9,
    top_k=200,
    max_tokens=5000,
    system_prompt="",
):
    bedrock_format = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": json.loads(messages),
        "temperature": temperature,
        "top_p": top_p
    }

    if system_prompt:
        bedrock_format["system"] = system_prompt

    response = await bedrock_runtime.invoke_model(body=json.dumps(bedrock_format), modelId=model_id)
    response_body = json.loads(response.get('body').read())

    return response_body

@retry_with_count(max_attempts=20, delay=10)
def inference_with_converse_api(bedrock_client,
                          model_id,
                          messages,
                          system_prompts="",
                          tools=[],
                          temperature=0.1,
                          top_p=0.9,
                          top_k=200,
                          max_tokens=1000,
                          thinking_config=None,
                          return_content=True
                         ):
    # Base inference parameters to use.
    inference_config = {"temperature": temperature}

    params = {
        "modelId": model_id,
        "messages": messages,
        "inferenceConfig": inference_config,
    }

    if tools:
        params["toolConfig"] = {"tools": tools}

    if system_prompts:
        params["system"] = [{"text": system_prompts}]

    if max_tokens:
        params["inferenceConfig"]["maxTokens"] = max_tokens

    # Note: top_p disabled - Claude Opus 4.5 does not support both temperature and top_p
    # if top_p:
    #     params["inferenceConfig"]["topP"] = top_p

    if top_k is not None and 'claude' in model_id:
        params["additionalModelRequestFields"] = {
            "top_k": top_k
        }

    if thinking_config:
        params["additionalModelRequestFields"]['thinking'] = thinking_config
        params["inferenceConfig"]["temperature"] = 1


    # Send the message with retry logic for ReadTimeoutError
    max_timeout_retries = 5
    for timeout_attempt in range(max_timeout_retries):
        try:
            response = bedrock_client.converse(**params)
            break  # Success, exit retry loop
        except ReadTimeoutError as e:
            if timeout_attempt < max_timeout_retries - 1:
                wait_time = (timeout_attempt + 1) * 30  # 30s, 60s, 90s, 120s
                logger.warning(f"ReadTimeoutError on attempt {timeout_attempt + 1}/{max_timeout_retries}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"ReadTimeoutError after {max_timeout_retries} attempts. Returning empty response.")
                return []  # Return empty list as empty response
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Non-timeout exception: {e}")
            return {}
    else:
        # All retries exhausted (should not reach here due to return in except block)
        return []

    if return_content:
        return response['output']['message']['content']
    else:
        return response


def build_message(texts=None, images=None, invoke_model=False):
    message = {
        "role": "user",
        "content": [],
    }

    if texts:
        if invoke_model:
            message["content"] += [{"text": text, "type": "text"} for text in texts]
        else:
            message["content"] += [{"text": text} for text in texts]
    if images:
        if invoke_model:
            message["content"] += [{"image": {
                "type": 'image',
                "source": {
                    "type": "base64",
                    "media_type":"image/jpeg",
                    "data": img
                }
            } for img in images}]
        else:
            message["content"] += [{"image": {
                "format": 'png',
                "source": {
                    "bytes": img
                }
            } for img in images}]

    return message

def get_json(contents, tool_name="print_pdf_content"):
    if not contents:
        return None

    json_classification = None
    for content in contents:
        if "toolUse" in content and content['toolUse']['name'] == tool_name:
            json_classification = content['toolUse']['input']
            break

    return json_classification
async def async_inference_with_converse_api(bedrock_client,
                          model_id,
                          messages,
                          system_prompts="",
                          tools=[],
                          temperature=0.1,
                          top_p=0.9,
                          top_k=200,
                          max_tokens=1000,
                          thinking_config=None
                         ):
    """Run inference with Bedrock Converse API with robust retry handling."""
    # Base inference parameters to use.
    inference_config = {"temperature": temperature}

    params = {
        "modelId": model_id,
        "messages": messages,
        "inferenceConfig": inference_config,
    }

    if tools:
        params["toolConfig"] = {"tools": tools}

    if system_prompts:
        params["system"] = [{"text": system_prompts}]

    # Additional inference parameters to use.
    if "nova" in model_id:
        # Send the message.
        async def make_api_call():
            with ThreadPoolExecutor(max_workers=1) as executor:
                return await asyncio.get_event_loop().run_in_executor(
                    executor, lambda: bedrock_client.converse(**params)
                )

        # Use the retry function with backoff
        response = await retry_async_with_backoff(make_api_call, max_attempts=10)
        if response is None:
            logger.error("Failed to get response after all retries, returning empty response")
            return []  # Return empty list as empty response

        return response['output']['message']['content']
    else:
        params["additionalModelRequestFields"] = {
            "max_tokens": max_tokens
        }
        if top_k is not None:
            params["additionalModelRequestFields"]['top_k'] = top_k
        elif top_p is not None and thinking_config is None:
            params["additionalModelRequestFields"]["top_p"] = top_p

        if thinking_config:
            params["additionalModelRequestFields"]['thinking'] = thinking_config
            logger.info("Temperature must be set to 1 and top_p must be unset")
            params["inferenceConfig"]["temperature"] = 1

        # Send the message asynchronously using a custom executor
        async def make_api_call():
            with ThreadPoolExecutor(max_workers=5) as executor:  # Increased max_workers
                return await asyncio.get_event_loop().run_in_executor(
                    executor, lambda: bedrock_client.converse(**params)
                )

        # Use the retry function with backoff
        response = await retry_async_with_backoff(make_api_call, max_attempts=10)
        logger.info(f"response: {response}")
        if response is None:
            logger.error("Failed to get response after all retries, returning empty response")
            return []  # Return empty list as empty response

        return response['output']['message']['content']

async def async_get_json(contents, tool_name="print_pdf_content"):
    if not contents:
        return None

    json_classification = None
    for content in contents:
        if "toolUse" in content and content['toolUse']['name'] == tool_name:
            json_classification = content['toolUse']['input']
            break

    return json_classification


# ============================================================================
# Bedrock Batch Inference for Embeddings
# ============================================================================

def create_bedrock_client(region_name: str = "us-west-2", config: Config = None) -> boto3.client:
    """Create a Bedrock Runtime client."""
    if config is None:
        config = boto_config
    return boto3.client("bedrock-runtime", region_name=region_name, config=config)


def prepare_batch_embedding_input(
    strings: list,
    model_id: str = "amazon.titan-embed-text-v2:0",
    output_path: str = None,
    embedding_dim: int = None
) -> str:
    """
    Prepare JSONL input file for Bedrock Batch Inference.

    Args:
        strings: List of strings to embed
        model_id: Embedding model ID
        output_path: Path to save JSONL file (optional, uses temp file if None)
        embedding_dim: Embedding dimension for Titan V2 (256, 384, 512, or 1024).
                       If None, uses model default (1024 for Titan V2).

    Returns:
        Path to the created JSONL file
    """
    import tempfile

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix='.jsonl', prefix='batch_embed_')
        os.close(fd)

    with open(output_path, 'w') as f:
        for idx, text in enumerate(strings):
            # Build model input based on model type
            if model_id == "amazon.titan-embed-text-v2:0" and embedding_dim is not None:
                model_input = {
                    "inputText": text,
                    "dimensions": embedding_dim,
                    "normalize": True,
                    "embeddingTypes": ["float"]
                }
            elif model_id.startswith("cohere.embed"):
                # Cohere models (v3, v4, multilingual, english)
                # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed.html
                model_input = {
                    "texts": [text],
                    "input_type": "clustering",
                    "truncate": "END"
                }
                # Cohere Embed v4 supports embedding_types parameter
                if "v4" in model_id:
                    model_input["embedding_types"] = ["float"]
            else:
                # Default: just inputText (for Titan V1 and others)
                model_input = {
                    "inputText": text
                }

            record = {
                "recordId": str(idx),
                "modelInput": model_input
            }
            f.write(json.dumps(record) + '\n')

    logger.info(f"Created batch input file with {len(strings)} records: {output_path}")
    return output_path


def upload_to_s3(local_path: str, bucket: str, key: str, region: str = "us-west-2") -> str:
    """
    Upload a file to S3.

    Args:
        local_path: Local file path
        bucket: S3 bucket name
        key: S3 object key
        region: AWS region

    Returns:
        S3 URI (s3://bucket/key)
    """
    s3_client = boto3.client('s3', region_name=region)
    s3_client.upload_file(local_path, bucket, key)
    s3_uri = f"s3://{bucket}/{key}"
    logger.info(f"Uploaded {local_path} to {s3_uri}")
    return s3_uri


def download_from_s3(s3_uri: str, local_path: str, region: str = "us-west-2") -> str:
    """
    Download a file from S3.

    Args:
        s3_uri: S3 URI (s3://bucket/key)
        local_path: Local destination path
        region: AWS region

    Returns:
        Local file path
    """
    s3_client = boto3.client('s3', region_name=region)

    # Parse S3 URI
    if s3_uri.startswith('s3://'):
        parts = s3_uri[5:].split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
    else:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")

    s3_client.download_file(bucket, key, local_path)
    logger.info(f"Downloaded {s3_uri} to {local_path}")
    return local_path


def list_s3_objects(s3_uri: str, region: str = "us-west-2") -> list:
    """
    List objects in S3 prefix.

    Args:
        s3_uri: S3 URI prefix (s3://bucket/prefix)
        region: AWS region

    Returns:
        List of S3 object keys
    """
    s3_client = boto3.client('s3', region_name=region)

    # Parse S3 URI
    if s3_uri.startswith('s3://'):
        parts = s3_uri[5:].split('/', 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ''
    else:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")

    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

    objects = []
    if 'Contents' in response:
        objects = [obj['Key'] for obj in response['Contents']]

    return objects


def create_batch_inference_job(
    input_s3_uri: str,
    output_s3_uri: str,
    model_id: str = "amazon.titan-embed-text-v2:0",
    role_arn: str = None,
    job_name: str = None,
    region: str = "us-west-2"
) -> dict:
    """
    Create a Bedrock Batch Inference job.

    Args:
        input_s3_uri: S3 URI for input JSONL file
        output_s3_uri: S3 URI prefix for output
        model_id: Model ID for embeddings
        role_arn: IAM role ARN with Bedrock and S3 permissions
        job_name: Optional job name (auto-generated if None)
        region: AWS region

    Returns:
        Job creation response with jobArn
    """
    import uuid

    bedrock_client = boto3.client('bedrock', region_name=region)

    if job_name is None:
        job_name = f"embedding-batch-{uuid.uuid4().hex[:8]}"

    if role_arn is None:
        # Try to get from environment or use a default pattern
        role_arn = os.environ.get('BEDROCK_BATCH_ROLE_ARN')
        if not role_arn:
            raise ValueError(
                "role_arn is required. Set BEDROCK_BATCH_ROLE_ARN environment variable "
                "or pass role_arn parameter."
            )

    params = {
        'jobName': job_name,
        'modelId': model_id,
        'roleArn': role_arn,
        'inputDataConfig': {
            's3InputDataConfig': {
                's3Uri': input_s3_uri
            }
        },
        'outputDataConfig': {
            's3OutputDataConfig': {
                's3Uri': output_s3_uri
            }
        }
    }

    response = bedrock_client.create_model_invocation_job(**params)
    logger.info(f"Created batch job: {response['jobArn']}")
    return response


def get_batch_job_status(job_arn: str, region: str = "us-west-2") -> dict:
    """
    Get the status of a Bedrock Batch Inference job.

    Args:
        job_arn: Job ARN from create_model_invocation_job
        region: AWS region

    Returns:
        Job status response
    """
    bedrock_client = boto3.client('bedrock', region_name=region)
    response = bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)
    return response


def wait_for_batch_job(
    job_arn: str,
    region: str = "us-west-2",
    poll_interval: int = 30,
    max_wait_time: int = 3600,
    show_progress: bool = True
) -> dict:
    """
    Wait for a Bedrock Batch Inference job to complete.

    Args:
        job_arn: Job ARN from create_model_invocation_job
        region: AWS region
        poll_interval: Seconds between status checks
        max_wait_time: Maximum seconds to wait
        show_progress: Whether to print progress updates

    Returns:
        Final job status response

    Raises:
        TimeoutError: If job doesn't complete within max_wait_time
        RuntimeError: If job fails
    """
    from tqdm import tqdm

    start_time = time.time()
    pbar = None

    if show_progress:
        pbar = tqdm(total=100, desc="Batch job progress", unit="%")
        last_progress = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait_time:
            if pbar:
                pbar.close()
            raise TimeoutError(f"Batch job did not complete within {max_wait_time} seconds")

        status = get_batch_job_status(job_arn, region)
        job_status = status.get('status', 'Unknown')

        # Update progress bar if available
        if pbar and 'inputDataConfig' in status:
            # Estimate progress based on status
            progress_map = {
                'Submitted': 10,
                'InProgress': 50,
                'Completed': 100,
                'Failed': 100,
                'Stopping': 80,
                'Stopped': 100,
                'PartiallyCompleted': 90,
                'Expired': 100,
                'Validating': 5
            }
            current_progress = progress_map.get(job_status, 0)
            if current_progress > last_progress:
                pbar.update(current_progress - last_progress)
                last_progress = current_progress

        if show_progress:
            logger.info(f"Job status: {job_status} (elapsed: {elapsed:.0f}s)")

        if job_status == 'Completed':
            if pbar:
                pbar.update(100 - last_progress)
                pbar.close()
            logger.info("Batch job completed successfully")
            return status

        if job_status in ['Failed', 'Stopped', 'Expired']:
            if pbar:
                pbar.close()
            error_msg = status.get('message', 'Unknown error')
            raise RuntimeError(f"Batch job {job_status}: {error_msg}")

        time.sleep(poll_interval)


def parse_batch_embedding_output(output_s3_uri: str, region: str = "us-west-2") -> dict:
    """
    Parse the output from a Bedrock Batch Inference job for embeddings.

    Args:
        output_s3_uri: S3 URI prefix where output was written
        region: AWS region

    Returns:
        Dictionary mapping recordId to embedding array
    """
    import tempfile
    import numpy as np

    # List output files
    output_files = list_s3_objects(output_s3_uri, region)

    embeddings = {}

    for obj_key in output_files:
        if not obj_key.endswith('.jsonl.out'):
            continue

        # Download and parse
        s3_uri = f"s3://{output_s3_uri.split('/')[2]}/{obj_key}"

        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            download_from_s3(s3_uri, tmp_path, region)

            with open(tmp_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue

                    record = json.loads(line)
                    record_id = record.get('recordId')
                    model_output = record.get('modelOutput', {})

                    # Titan embedding output format
                    embedding = model_output.get('embedding')
                    if embedding:
                        embeddings[record_id] = np.array(embedding)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    logger.info(f"Parsed {len(embeddings)} embeddings from batch output")
    return embeddings


def batch_compute_embeddings(
    strings: list,
    model_id: str = "amazon.titan-embed-text-v2:0",
    s3_bucket: str = None,
    s3_prefix: str = "bedrock-batch/embeddings",
    role_arn: str = None,
    region: str = "us-west-2",
    poll_interval: int = 30,
    max_wait_time: int = 3600,
    show_progress: bool = True,
    cleanup: bool = True,
    min_records: int = 100,
    embedding_dim: int = None
) -> dict:
    """
    Compute embeddings for a list of strings using Bedrock Batch Inference.

    This is the main entry point for true batch embedding computation.
    It handles the full workflow: prepare input, upload to S3, create job,
    wait for completion, download and parse results.

    IMPORTANT: Bedrock Batch Inference requires a minimum of 100 records.
    For smaller datasets, use parallel API calls instead.

    Args:
        strings: List of strings to embed (minimum 100 required)
        model_id: Embedding model ID
        s3_bucket: S3 bucket for input/output (required)
        s3_prefix: S3 prefix for files
        role_arn: IAM role ARN with Bedrock and S3 permissions
        region: AWS region
        poll_interval: Seconds between status checks
        max_wait_time: Maximum seconds to wait for job
        show_progress: Whether to show progress
        cleanup: Whether to clean up temporary files
        min_records: Minimum records required (default 100, Bedrock requirement)
        embedding_dim: Embedding dimension for Titan V2 (256, 384, 512, or 1024).
                       If None, uses model default (1024 for Titan V2).

    Returns:
        Dictionary mapping original string -> np.ndarray embedding

    Raises:
        ValueError: If fewer than min_records strings provided
    """
    import tempfile
    import uuid

    if s3_bucket is None:
        s3_bucket = os.environ.get('BEDROCK_BATCH_S3_BUCKET')
        if not s3_bucket:
            raise ValueError(
                "s3_bucket is required. Set BEDROCK_BATCH_S3_BUCKET environment variable "
                "or pass s3_bucket parameter."
            )

    if not strings:
        return {}

    if len(strings) < min_records:
        raise ValueError(
            f"Bedrock Batch Inference requires at least {min_records} records, "
            f"but only {len(strings)} provided. Use parallel API calls for smaller datasets."
        )

    job_id = uuid.uuid4().hex[:8]

    # Step 1: Prepare input JSONL
    if show_progress:
        logger.info(f"Preparing batch input for {len(strings)} strings...")

    input_path = prepare_batch_embedding_input(strings, model_id, embedding_dim=embedding_dim)

    try:
        # Step 2: Upload to S3
        input_s3_key = f"{s3_prefix}/input/{job_id}/input.jsonl"
        input_s3_uri = upload_to_s3(input_path, s3_bucket, input_s3_key, region)

        output_s3_prefix = f"{s3_prefix}/output/{job_id}/"
        output_s3_uri = f"s3://{s3_bucket}/{output_s3_prefix}"

        # Step 3: Create batch job
        job_response = create_batch_inference_job(
            input_s3_uri=input_s3_uri,
            output_s3_uri=output_s3_uri,
            model_id=model_id,
            role_arn=role_arn,
            job_name=f"embed-{job_id}",
            region=region
        )
        job_arn = job_response['jobArn']

        # Step 4: Wait for completion
        wait_for_batch_job(
            job_arn=job_arn,
            region=region,
            poll_interval=poll_interval,
            max_wait_time=max_wait_time,
            show_progress=show_progress
        )

        # Step 5: Parse results
        embeddings_by_id = parse_batch_embedding_output(output_s3_uri, region)

        # Map back to original strings
        result = {}
        for idx, text in enumerate(strings):
            record_id = str(idx)
            if record_id in embeddings_by_id:
                result[text] = embeddings_by_id[record_id]

        logger.info(f"Successfully computed {len(result)} embeddings via batch inference")
        return result

    finally:
        # Cleanup local temp file
        if cleanup and os.path.exists(input_path):
            os.remove(input_path)


def batch_compute_embeddings_chunked(
    strings: list,
    model_id: str = "amazon.titan-embed-text-v2:0",
    s3_bucket: str = None,
    s3_prefix: str = "bedrock-batch/embeddings",
    role_arn: str = None,
    region: str = "us-west-2",
    chunk_size: int = 50000,
    poll_interval: int = 30,
    max_wait_time: int = 7200,
    show_progress: bool = True,
    embedding_dim: int = None
) -> dict:
    """
    Compute embeddings with automatic chunking for very large datasets.

    Bedrock Batch Inference has limits on input file size. This function
    automatically chunks large string lists and runs multiple batch jobs.

    Args:
        strings: List of strings to embed
        model_id: Embedding model ID
        s3_bucket: S3 bucket for input/output
        s3_prefix: S3 prefix for files
        role_arn: IAM role ARN
        region: AWS region
        chunk_size: Maximum strings per batch job
        poll_interval: Seconds between status checks
        max_wait_time: Maximum seconds to wait per job
        show_progress: Whether to show progress
        embedding_dim: Embedding dimension for Titan V2 (256, 384, 512, or 1024).
                       If None, uses model default (1024 for Titan V2).

    Returns:
        Dictionary mapping string -> np.ndarray embedding
    """
    from tqdm import tqdm

    if not strings:
        return {}

    # Remove duplicates while preserving order
    unique_strings = list(dict.fromkeys(strings))

    all_embeddings = {}
    num_chunks = (len(unique_strings) + chunk_size - 1) // chunk_size

    if show_progress:
        logger.info(f"Processing {len(unique_strings)} unique strings in {num_chunks} batch(es)")

    for i in range(0, len(unique_strings), chunk_size):
        chunk = unique_strings[i:i + chunk_size]
        chunk_num = i // chunk_size + 1

        if show_progress:
            logger.info(f"Processing chunk {chunk_num}/{num_chunks} ({len(chunk)} strings)")

        chunk_embeddings = batch_compute_embeddings(
            strings=chunk,
            model_id=model_id,
            s3_bucket=s3_bucket,
            s3_prefix=f"{s3_prefix}/chunk_{chunk_num}",
            role_arn=role_arn,
            region=region,
            poll_interval=poll_interval,
            max_wait_time=max_wait_time,
            show_progress=show_progress,
            embedding_dim=embedding_dim
        )

        all_embeddings.update(chunk_embeddings)

    return all_embeddings
