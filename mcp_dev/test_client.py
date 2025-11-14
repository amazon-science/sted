from mcp import stdio_client, StdioServerParameters
from strands import Agent
from strands.tools.mcp import MCPClient

# Connect to an MCP server using stdio transport
# Note: uvx command syntax differs by platform

# Create MCP client
mcp_client = MCPClient(lambda: stdio_client(
    StdioServerParameters(
        command="python", 
        args=["server.py"]
    )
))

# Manual lifecycle management
with mcp_client:
    # Get the tools from the MCP server
    tools = mcp_client.list_tools_sync()

    # Create an agent with these tools
    agent = Agent(tools=tools)
    agent("generate two json format variables with the same schema and calculate the similarity")  # Must be within context