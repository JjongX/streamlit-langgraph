"""
Example MCP Weather Server (supports both stdio and HTTP/SSE transport).

Run this server with:
    # Using FastMCP CLI (stdio transport):
    fastmcp run weather_server.py
    
    # Using FastMCP CLI with HTTP transport (for type="response", use 0.0.0.0 for public access):
    fastmcp run weather_server.py --transport http --host 0.0.0.0 --port 8000
    
Then configure it in your agent's mcp_servers configuration:
    # For stdio:
    "weather": {
        "transport": "stdio",
        "command": "python",
        "args": ["/path/to/weather_server.py"]
    }
    
    # For HTTP/SSE (after starting server):
    "weather": {
        "transport": "streamable_http",  # or "sse" for legacy SSE transport
        "url": "http://<ip-address>:8000/mcp"  # Use public IP for openai response api
    }
"""

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



