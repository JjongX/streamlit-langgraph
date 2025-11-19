"""
Example MCP Math Server (supports both stdio and HTTP/SSE transport).

stdio, http, sse, websocket

Run this server with:
    # Using FastMCP CLI (stdio transport):
    fastmcp run math_server.py
    
    # Using FastMCP CLI with HTTP transport (for type="response", use 0.0.0.0 for public access):
    fastmcp run math_server.py --transport http --host 0.0.0.0 --port 8000

Then configure it in your agent's mcp_servers configuration:
    # For stdio:
    "math": {
        "transport": "stdio",
        "command": "python",
        "args": ["/path/to/math_server.py"]
    }
    
    # For HTTP/SSE (after starting server):
    "math": {
        "transport": "streamable_http",  # or "sse" for legacy SSE transport
        "url": "http://<ip-address>:8000/mcp"  # Use public IP for openai response api
    }
"""

from fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract b from a"""
    return a - b

@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
