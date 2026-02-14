#!/usr/bin/env python3
"""Test Minimax-M2 response handling"""
import boto3
import json
import re

def extract_json_from_string(input_string):
    """Extract JSON from string"""
    input_string = re.sub(r'^```(?:json)?\s*\n?', '', input_string.strip())
    input_string = re.sub(r'\n?```\s*$', '', input_string)
    match = re.search(r'[{[]', input_string)
    if not match:
        raise ValueError('No JSON data found in the string')
    start_pos = match.start()
    start_char = match.group()
    end_char = '}' if start_char == '{' else ']'
    count = 0
    end_pos = -1
    for i in range(start_pos, len(input_string)):
        if input_string[i] == start_char:
            count += 1
        elif input_string[i] == end_char:
            count -= 1
            if count == 0:
                end_pos = i
                break
    if end_pos == -1:
        raise ValueError('No matching closing bracket/brace found')
    json_string = input_string[start_pos:end_pos + 1]
    return json.loads(json_string)


# Create Bedrock client
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

# Test prompt
messages = [
    {
        'role': 'user',
        'content': [{'text': 'Return a JSON object with a greeting field. Example: {"greeting": "hello"}'}]
    }
]

print('Testing Minimax-M2...')
print()

response = bedrock.converse(
    modelId='minimax.minimax-m2',
    messages=messages,
    inferenceConfig={
        'maxTokens': 500,
        'temperature': 0.0
    }
)

content = response['output']['message']['content']
print(f'Raw content (type={type(content)}):')
print(json.dumps(content, indent=2))
print()

# Test the extraction logic from generate_structured_outputs.py
print('=== Testing extraction logic ===')
response_text = content[0].get('text', '{}')
print(f'Step 1 - First element text: {repr(response_text)}')

if response_text == '{}' and len(content) > 1:
    print('Step 2 - Entering fallback loop...')
    for item in content:
        print(f'  Checking item keys: {list(item.keys())}')
        if isinstance(item, dict) and 'text' in item:
            response_text = item['text']
            print(f'  Found text!')
            break

print(f'Final response_text: {repr(response_text)}')
print()

# Try JSON extraction
print('=== Testing JSON extraction ===')
try:
    parsed = extract_json_from_string(response_text)
    print(f'SUCCESS! Parsed JSON: {parsed}')
except Exception as e:
    print(f'FAILED: {e}')
