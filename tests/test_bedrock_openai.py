"""
Test tool calling using OpenRouter API with GPT-4.
This demonstrates that OpenAI API has no 64-character tool name limit.

FINDING: The 64-character tool name limit is specific to AWS Bedrock infrastructure.
- Bedrock (both Converse API and invoke_model) enforces: ^[a-zA-Z0-9_-]{1,64}$
- OpenRouter routes Claude through Bedrock, so it also has this limit
- To bypass: Use Anthropic's direct API (api.anthropic.com) with ANTHROPIC_API_KEY
- OpenAI API has no such limitation
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

# Load from .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 1. Define Tool with long name (69 characters - exceeds Bedrock's 64 limit)
@tool
def office_word_document_server_convert_footnotes_to_endnotes_in_document(document_id: str):
    """Converts all footnotes in a Word document to endnotes."""
    return f"Converted footnotes to endnotes in document {document_id}"

tools = [office_word_document_server_convert_footnotes_to_endnotes_in_document]

# 2. Initialize OpenRouter Model with GPT-4 (not routed through Bedrock)
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

if not api_key:
    print("Error: OPENAI_API_KEY not set. Please set it to your OpenRouter API key.")
    exit(1)

# Use GPT-4 instead of Claude to avoid Bedrock routing
model = ChatOpenAI(
    model="openai/gpt-4-turbo",  # OpenRouter model name for OpenAI
    temperature=0,
    api_key=api_key,
    base_url=base_url,
)

# Bind tools to the model
model_with_tools = model.bind_tools(tools)

# 3. Test tool calling with long tool name
print("=" * 60)
print("Testing OpenRouter API with GPT-4 Tool Calling")
print("(OpenAI API has no 64-character tool name limit)")
print("=" * 60)

tool_name = tools[0].name
print(f"\nTool name length: {len(tool_name)} characters")
print(f"Tool name: {tool_name}")
print(f"Base URL: {base_url}")

print("\nSending message to GPT-4...")

try:
    messages = [HumanMessage(content="Convert all footnotes to endnotes in document abc123")]
    response = model_with_tools.invoke(messages)

    print(f"\nResponse type: {type(response)}")
    print(f"Response content: {response.content}")

    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"\nTool calls detected: {len(response.tool_calls)}")
        for tc in response.tool_calls:
            print(f"  - Tool: {tc['name']}")
            print(f"    Args: {tc['args']}")
    else:
        print("\nNo tool calls in response")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
