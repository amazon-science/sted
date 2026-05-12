"""Bedrock embedding backends for STED.

Extracted from semantic_json_tree_consistency.py during the v0.2.0 refactor.
The original methods are thin wrappers that delegate to these module-level
helpers, so existing call sites and import paths keep working.
"""
from __future__ import annotations

import asyncio
import json
import os as _os
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from ..utils import get_embeddings


def batch_encode_bedrock(
    evaluator,
    strings: List[str],
    show_progress: bool = True,
    max_workers: int = 10,
) -> None:
    """Batch encode strings using Bedrock API with parallel calls."""
    def encode_single(text: str) -> Tuple[str, Optional[np.ndarray]]:
        try:
            emb = get_embeddings(
                text,
                evaluator.model_id,
                evaluator.bedrock_client,
                output_embedding_length=evaluator.embedding_dim,
            )
            return (text, emb)
        except Exception as e:
            warnings.warn(f"Bedrock embedding failed for '{text[:50]}...': {e}")
            return (text, None)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(encode_single, s): s for s in strings}

        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(strings), desc="Bedrock embeddings")

        for future in iterator:
            text, emb = future.result()
            if emb is not None:
                evaluator._store_embedding(text, emb)


def batch_encode_bedrock_async(
    evaluator,
    strings: List[str],
    show_progress: bool = True,
    max_concurrent: int = 50,
    region_name: str = None,
) -> None:
    """
    Batch encode strings using Bedrock API with async calls.

    This method uses aioboto3 for true async I/O, which is more efficient than
    ThreadPoolExecutor for I/O-bound operations like API calls.

    Args:
        evaluator: SemanticJsonTreeConsistencyEvaluator instance
        strings: List of strings to embed
        show_progress: Whether to show progress bar
        max_concurrent: Maximum concurrent API calls (default 50)
        region_name: AWS region name (default: use client's region)
    """
    try:
        import aioboto3
    except ImportError:
        warnings.warn("aioboto3 not installed. Falling back to ThreadPoolExecutor. "
                     "Install with: pip install aioboto3")
        batch_encode_bedrock(evaluator, strings, show_progress, max_workers=max_concurrent)
        return

    if region_name is None:
        if hasattr(evaluator.bedrock_client, '_client_config'):
            region_name = evaluator.bedrock_client._client_config.region_name
        else:
            region_name = "us-west-2"

    async def encode_single_async(
        text: str,
        client,
        semaphore: asyncio.Semaphore
    ) -> Tuple[str, Optional[np.ndarray]]:
        """Encode a single string asynchronously."""
        async with semaphore:
            try:
                # Build request body based on model
                if evaluator.model_id == "amazon.titan-embed-text-v1":
                    request_body = {"inputText": text}
                elif evaluator.model_id == "amazon.titan-embed-text-v2:0":
                    request_body = {
                        "inputText": text,
                        "dimensions": evaluator.embedding_dim,
                        "normalize": True,
                        "embeddingTypes": ["float"]
                    }
                elif evaluator.model_id == "cohere.embed-multilingual-v3":
                    # Cohere Embed Multilingual V3 returns fixed 1024-dim embeddings
                    request_body = {
                        "texts": [text],
                        "input_type": "clustering",
                        "truncate": "END"
                    }
                elif evaluator.model_id in ["cohere.embed-v4:0", "us.cohere.embed-v4:0"]:
                    request_body = {
                        "texts": [text],
                        "input_type": "search_document",
                        "embedding_types": ["float"]
                    }
                else:
                    request_body = {"inputText": text}

                # Call Bedrock API
                response = await client.invoke_model(
                    modelId=evaluator.model_id,
                    body=json.dumps(request_body),
                    contentType="application/json",
                    accept="application/json"
                )

                # Read response body
                response_body = await response['body'].read()
                result = json.loads(response_body)

                # Extract embedding based on model
                if evaluator.model_id == "cohere.embed-multilingual-v3":
                    embedding = np.array(result["embeddings"][0])
                elif evaluator.model_id in ["cohere.embed-v4:0", "us.cohere.embed-v4:0"]:
                    # Cohere Embed V4 returns embeddings as dict with 'float' key
                    embedding = np.array(result["embeddings"]["float"][0])
                elif "embeddingsByType" in result:
                    embedding = np.array(result["embeddingsByType"]["float"])
                else:
                    embedding = np.array(result.get("embedding", []))

                return (text, embedding)

            except Exception as e:
                warnings.warn(f"Async Bedrock embedding failed for '{text[:50]}...': {e}")
                return (text, None)

    async def encode_all_async():
        """Encode all strings concurrently."""
        session = aioboto3.Session()
        semaphore = asyncio.Semaphore(max_concurrent)

        async with session.client('bedrock-runtime', region_name=region_name) as client:
            tasks = [encode_single_async(s, client, semaphore) for s in strings]

            if show_progress:
                results = []
                for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks),
                                  desc="Async Bedrock embeddings"):
                    result = await coro
                    results.append(result)
            else:
                results = await asyncio.gather(*tasks)

            return results

    # Run the async function
    try:
        # Check if we're already in an event loop
        loop = asyncio.get_running_loop()
        # If we are, use nest_asyncio or run in executor
        import nest_asyncio
        nest_asyncio.apply()
        results = asyncio.run(encode_all_async())
    except RuntimeError:
        # No running event loop, safe to use asyncio.run
        results = asyncio.run(encode_all_async())

    # Store results in embedding dict
    for text, emb in results:
        if emb is not None:
            evaluator._embedding_dict[text] = emb


def batch_encode_bedrock_auto(
    evaluator,
    strings: List[str],
    show_progress: bool = True,
    max_workers: int = 10,
    s3_bucket: str = None,
    s3_prefix: str = "bedrock-batch/embeddings",
    role_arn: str = None,
    batch_threshold: int = 5000,
) -> None:
    """
    Auto-select the best embedding method based on string count.

    This method automatically chooses between parallel API calls and
    batch inference based on the number of strings to embed:

    - < 100 strings: Parallel API (batch inference not supported)
    - 100 - batch_threshold: Parallel API (faster, no job overhead)
    - > batch_threshold: Batch inference (no rate limits, better throughput)

    Args:
        evaluator: SemanticJsonTreeConsistencyEvaluator instance
        strings: List of strings to embed
        show_progress: Whether to show progress
        max_workers: Max parallel workers for API calls
        s3_bucket: S3 bucket for batch inference
        s3_prefix: S3 prefix for batch inference files
        role_arn: IAM role ARN for batch inference
        batch_threshold: String count threshold for batch inference (default 5000)
    """
    num_strings = len(strings)

    # Check if batch inference is configured
    batch_configured = (s3_bucket is not None or
                       _os.environ.get('BEDROCK_BATCH_S3_BUCKET') is not None)

    if num_strings < 100:
        # Batch inference not supported below 100
        if show_progress:
            print(f"Using parallel API calls ({num_strings} strings < 100 minimum for batch)")
        batch_encode_bedrock(evaluator, strings, show_progress, max_workers)

    elif num_strings <= batch_threshold or not batch_configured:
        # Parallel API is faster for medium-sized datasets
        if show_progress:
            if not batch_configured:
                print(f"Using parallel API calls ({num_strings} strings, batch not configured)")
            else:
                print(f"Using parallel API calls ({num_strings} strings <= {batch_threshold} threshold)")
        batch_encode_bedrock(evaluator, strings, show_progress, max_workers)

    else:
        # Large dataset - use batch inference with async parallel chunks
        if show_progress:
            print(f"Using batch inference ({num_strings} strings > {batch_threshold} threshold)")
        batch_encode_bedrock_batch_inference_async(
            evaluator, strings, show_progress,
            s3_bucket=s3_bucket, s3_prefix=s3_prefix, role_arn=role_arn
        )


def batch_encode_bedrock_batch_inference_async(
    evaluator,
    strings: List[str],
    show_progress: bool = True,
    s3_bucket: str = None,
    s3_prefix: str = "bedrock-batch/embeddings",
    role_arn: str = None,
    chunk_size: int = 50000,
) -> None:
    """
    Batch encode using async parallel batch inference jobs.

    For very large datasets, this method chunks the strings and runs
    multiple batch inference jobs concurrently using asyncio.

    Args:
        evaluator: SemanticJsonTreeConsistencyEvaluator instance
        strings: List of strings to embed
        show_progress: Whether to show progress
        s3_bucket: S3 bucket for batch inference
        s3_prefix: S3 prefix for files
        role_arn: IAM role ARN for batch inference
        chunk_size: Strings per batch job (default 50000)
    """
    from ..bedrock_utils import (
        prepare_batch_embedding_input,
        upload_to_s3,
        create_batch_inference_job,
        get_batch_job_status,
        parse_batch_embedding_output
    )

    # Get configuration
    if s3_bucket is None:
        s3_bucket = _os.environ.get('BEDROCK_BATCH_S3_BUCKET')
    if role_arn is None:
        role_arn = _os.environ.get('BEDROCK_BATCH_ROLE_ARN')

    region = "us-west-2"
    if hasattr(evaluator.bedrock_client, '_client_config'):
        region = evaluator.bedrock_client._client_config.region_name

    # Chunk the strings
    chunks = [strings[i:i + chunk_size] for i in range(0, len(strings), chunk_size)]
    num_chunks = len(chunks)

    if show_progress:
        print(f"Splitting {len(strings)} strings into {num_chunks} batch job(s)...")

    async def submit_and_wait_job(chunk_strings: List[str], chunk_idx: int) -> dict:
        """Submit a batch job and wait for completion."""
        job_id = uuid.uuid4().hex[:8]

        # Prepare input
        input_path = prepare_batch_embedding_input(chunk_strings, evaluator.model_id, embedding_dim=evaluator.embedding_dim)

        try:
            # Upload to S3
            input_s3_key = f"{s3_prefix}/input/{job_id}/input.jsonl"
            input_s3_uri = upload_to_s3(input_path, s3_bucket, input_s3_key, region)

            output_s3_prefix = f"{s3_prefix}/output/{job_id}/"
            output_s3_uri = f"s3://{s3_bucket}/{output_s3_prefix}"

            # Create batch job
            job_response = create_batch_inference_job(
                input_s3_uri=input_s3_uri,
                output_s3_uri=output_s3_uri,
                model_id=evaluator.model_id,
                role_arn=role_arn,
                job_name=f"embed-chunk{chunk_idx}-{job_id}",
                region=region
            )
            job_arn = job_response['jobArn']

            if show_progress:
                print(f"  Chunk {chunk_idx + 1}/{num_chunks}: Job submitted ({len(chunk_strings)} strings)")

            # Poll for completion (async sleep)
            while True:
                status = get_batch_job_status(job_arn, region)
                job_status = status.get('status', 'Unknown')

                if job_status == 'Completed':
                    # Parse results
                    embeddings = parse_batch_embedding_output(output_s3_uri, region)

                    # Map back to original strings
                    result = {}
                    for idx, text in enumerate(chunk_strings):
                        record_id = str(idx)
                        if record_id in embeddings:
                            result[text] = embeddings[record_id]

                    if show_progress:
                        print(f"  Chunk {chunk_idx + 1}/{num_chunks}: Completed ({len(result)} embeddings)")
                    return result

                if job_status in ['Failed', 'Stopped', 'Expired']:
                    error_msg = status.get('message', 'Unknown error')
                    warnings.warn(f"Chunk {chunk_idx + 1} failed: {error_msg}")
                    return {}

                await asyncio.sleep(15)  # Non-blocking sleep

        finally:
            # Cleanup local temp file
            if _os.path.exists(input_path):
                _os.remove(input_path)

    async def run_all_jobs():
        """Run all batch jobs concurrently."""
        tasks = [submit_and_wait_job(chunk, idx) for idx, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_embeddings = {}
        for result in results:
            if isinstance(result, dict):
                all_embeddings.update(result)
            elif isinstance(result, Exception):
                warnings.warn(f"Batch job failed: {result}")

        return all_embeddings

    # Run async jobs
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # If already in async context, create task
        import nest_asyncio
        nest_asyncio.apply()
        embeddings = loop.run_until_complete(run_all_jobs())
    else:
        embeddings = loop.run_until_complete(run_all_jobs())

    # Update embedding dict
    evaluator._embedding_dict.update(embeddings)

    if show_progress:
        print(f"Batch inference completed: {len(embeddings)} total embeddings")


def batch_encode_bedrock_batch_inference(
    evaluator,
    strings: List[str],
    show_progress: bool = True,
    s3_bucket: str = None,
    s3_prefix: str = "bedrock-batch/embeddings",
    role_arn: str = None,
    min_records: int = 100,
) -> None:
    """
    Batch encode strings using true Bedrock Batch Inference (S3-based async).

    This method uses Bedrock's batch inference API which is more efficient
    for large datasets (>100 strings) compared to parallel single API calls.

    IMPORTANT: Bedrock Batch Inference requires a minimum of 100 records.
    For smaller datasets, this method will automatically fall back to
    parallel API calls.

    Args:
        evaluator: SemanticJsonTreeConsistencyEvaluator instance
        strings: List of strings to embed
        show_progress: Whether to show progress
        s3_bucket: S3 bucket for input/output (required for batch inference)
        s3_prefix: S3 prefix for files
        role_arn: IAM role ARN for Bedrock Batch Inference
        min_records: Minimum records for batch inference (default 100)
    """
    # Check minimum records requirement
    if len(strings) < min_records:
        if show_progress:
            print(f"Only {len(strings)} strings (< {min_records} minimum). "
                  f"Using parallel API calls instead of batch inference.")
        batch_encode_bedrock(evaluator, strings, show_progress)
        return

    from ..bedrock_utils import batch_compute_embeddings_chunked

    if show_progress:
        print(f"Using Bedrock Batch Inference for {len(strings)} strings...")

    try:
        # Get region from bedrock client if possible
        region = "us-west-2"
        if hasattr(evaluator.bedrock_client, '_client_config'):
            region = evaluator.bedrock_client._client_config.region_name

        embeddings = batch_compute_embeddings_chunked(
            strings=strings,
            model_id=evaluator.model_id,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            role_arn=role_arn,
            region=region,
            show_progress=show_progress,
            embedding_dim=evaluator.embedding_dim
        )

        # Update embedding dict
        evaluator._embedding_dict.update(embeddings)

        if show_progress:
            print(f"Batch inference completed: {len(embeddings)} embeddings computed")

    except Exception as e:
        warnings.warn(f"Batch inference failed: {e}. Falling back to parallel API calls.")
        # Fallback to parallel single API calls
        batch_encode_bedrock(evaluator, strings, show_progress)
