"""
Example MCP Weather Server (supports both stdio and HTTP/SSE transport).

Run this server with:
    # Using FastMCP CLI:
    fastmcp run weather_server.py
    
    # Using FastMCP CLI with HTTP transport:
    fastmcp run weather_server.py --transport http
    
Then configure it in your agent's mcp_servers configuration:
    # For stdio:
    "weather": {
        "transport": "stdio",
        "command": "python",
        "args": ["/path/to/weather_server.py"]
    }
    
    # For HTTP/SSE (after starting server):
    "weather": {
        "transport": "http",  # or "sse" for legacy SSE transport
        "url": "http://127.0.0.1:8000/mcp"  # Update port if using --port option
    }
"""

import argparse
from fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for a location."""
    # This is a mock implementation
    # In a real scenario, you would call a weather API
    return f"The weather in {location} is sunny with a temperature of 72°F."

@mcp.tool()
async def get_forecast(location: str, days: int = 3) -> str:
    """Get weather forecast for a location."""
    return f"Forecast for {location} for the next {days} days: Mostly sunny, high 72°F, low 55°F."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Weather Server")
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



