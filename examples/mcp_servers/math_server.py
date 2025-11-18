"""
Example MCP Math Server (supports both stdio and HTTP/SSE transport).

Run this server with:
    # Using FastMCP CLI:
    fastmcp run math_server.py
    
    # Using FastMCP CLI with HTTP transport:
    fastmcp run math_server.py --transport http

Then configure it in your agent's mcp_servers configuration:
    # For stdio:
    "math": {
        "transport": "stdio",
        "command": "python",
        "args": ["/path/to/math_server.py"]
    }
    
    # For HTTP/SSE (after starting server):
    "math": {
        "transport": "http",  # or "sse" for legacy SSE transport
        "url": "http://127.0.0.1:8000/mcp"  # Update port if using --port option
    }
"""

import argparse

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

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="MCP Math Server")
    parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=["stdio", "http", "sse"],
        help="Transport type: stdio (default), http, or sse"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number for HTTP/SSE transport (default: 8000)"
    )
    
    args = parser.parse_args()
    transport = args.transport
    port = args.port
    
    mcp.run(transport=transport, host="127.0.0.1", port=port)
