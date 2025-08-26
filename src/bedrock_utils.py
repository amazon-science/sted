import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
from functools import wraps
import time
import asyncio
import random
from concurrent.futures import ThreadPoolExecutor
import boto3
from botocore.config import Config
import json

logger = logging.getLogger(__name__)

# Configure boto3 with retry settings
boto_config = Config(
    retries={
        'max_attempts': 20,
        'mode': 'adaptive'
    },
    max_pool_connections=50  # Increase connection pool size
)

sm_client = boto3.client("runtime.sagemaker", region_name="us-west-2")

def retry_with_count(max_attempts=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    print(f"Attempt {attempt + 1}/{max_attempts}")
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        #raise e
                        print(f'Error: {str(e)}')
                        return None
                    print(f"Attempt {attempt + 1} failed: {str(e)}")
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
            
            if "ThrottlingException" in str(e) or "TooManyRequestsException" in str(e):
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
    
    print(f"model_id: {model_id}")
    
    if tools:
        params["toolConfig"] = {"tools": tools}
    
    if system_prompts:
        params["system"] = [{"text": system_prompts}]
    
    if max_tokens:
        params["inferenceConfig"]["maxTokens"] = max_tokens
    
    if top_p:
        params["inferenceConfig"]["topP"] = top_p
    
    if top_k is not None and 'claude' in model_id:
        params["additionalModelRequestFields"] = {
            "top_k": top_k
        }
        
    if thinking_config:
        params["additionalModelRequestFields"]['thinking'] = thinking_config
        print(f"Temperature must be set to 1 and top_p must be unset")
        params["inferenceConfig"]["temperature"] = 1
        

    # Send the message.
    try:
        response = bedrock_client.converse(**params)
    except:
        import traceback
        traceback.print_exc()
        return {}
    
    if return_content:
        return response['output']['message']['content']
    else:
        return response
    
def inference_with_sm_endpoint(prompt, endpoint="jumpstart-dft-hf-reasoning-qwen3-32-20250820-053044", max_new_tokens=1000):
    body = {
        "inputs": f"<|begin_of_sentence|><|User|>{prompt}<|Assistant|>",
        "parameters": {
            "max_new_tokens": max_new_tokens
        }
    }
    
    response = client.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/json",
        Body=json.dumps(body),
        Accept="application/json"
    )
    
    result = json.loads(response["Body"].read().decode("utf-8"))
    return result["generated_text"]

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
            logger.error("Failed to get response after all retries")
            return None
        
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
            logger.error("Failed to get response after all retries")
            return None

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
