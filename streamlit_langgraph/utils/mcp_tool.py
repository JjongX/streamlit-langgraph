# MCP (Model Context Protocol) tool integration utilities.

import asyncio
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient


class MCPToolManager:
    """
    Manager for MCP (Model Context Protocol) tools.
    
    Handles loading tools from MCP servers and converting them to LangChain tools.
    Supports multiple MCP servers with different transport types (stdio, streamable_http, SSE).
    """
    
    def __init__(self):
        self.mcp_client: Optional[MultiServerMCPClient] = None
        self._server_configs: Dict[str, Dict[str, Any]] = {}
    
    def add_servers(self, servers: Dict[str, Dict[str, Any]]) -> None:
        """Add multiple MCP server configurations at once."""
        for server_name, config in servers.items():
            self._server_configs[server_name] = config
        self.mcp_client = None
    
    def _get_client(self) -> MultiServerMCPClient:
        """Get or create MCP client with configured servers."""
        if self.mcp_client is None:
            if not self._server_configs:
                raise ValueError("No MCP servers configured. Use add_servers() first.")
            self.mcp_client = MultiServerMCPClient(self._server_configs)
        
        return self.mcp_client
    
    async def get_tools_async(self) -> List[Any]:
        """Get all tools from all configured MCP servers (async)."""
        if not self._server_configs:
            return []
        
        client = self._get_client()
        tools = await client.get_tools()
        return tools
    
    def get_tools(self) -> List[Any]:
        """
        Get all tools from all configured MCP servers (sync wrapper).
        
        Returns:
            List of LangChain tools from MCP servers, wrapped for sync invocation
        
        Note:
            MCP tools are async-only, so they are wrapped to support sync invocation
            using asyncio.run() when the tool is called.
        """
        if not self._server_configs:
            return []
        
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import warnings
            warnings.warn(
                "Cannot load MCP tools synchronously when event loop is running. "
                "Consider using get_tools_async() or configuring MCP tools before starting the event loop."
            )
            return []
        async_tools = loop.run_until_complete(self.get_tools_async())
        
        return [self._wrap_async_tool(tool) for tool in async_tools]
    
    def _wrap_async_tool(self, async_tool: Any) -> StructuredTool:
        """Wrap an async tool to support sync invocation."""
        def sync_wrapper(**kwargs):
            """Sync wrapper that runs the async tool."""
            async def run_async():
                return await async_tool.ainvoke(kwargs)
            
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result = new_loop.run_until_complete(run_async())
                    new_loop.close()
                    return result
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    return future.result()
            else:
                return loop.run_until_complete(run_async())
        
        # Create a new StructuredTool with the sync wrapper
        # Preserve the original tool's args_schema if available
        args_schema = None
        if hasattr(async_tool, 'args_schema') and async_tool.args_schema:
            args_schema = async_tool.args_schema
        
        return StructuredTool.from_function(
            func=sync_wrapper,
            name=async_tool.name,
            description=async_tool.description,
            args_schema=args_schema,
        )
    
    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Get MCP tools in OpenAI Responses API format."""
        if not self._server_configs:
            return []
        
        tools = []
        for server_name, config in self._server_configs.items():
            transport = config.get("transport", "stdio")
            
            tool_dict = {
                "type": "mcp",
                "server_label": server_name,
                "server_description": config.get("description", f"MCP server: {server_name}"),
                "require_approval": config.get("require_approval", "never"),
            }
            
            # Handle HTTP-based transports (streamable_http, http, sse)
            if transport in ("streamable_http", "http", "sse"):
                if "url" in config:
                    url = config["url"]
                    # Validate URL is publicly accessible (not localhost) for Responses API
                    if "localhost" in url or "127.0.0.1" in url:
                        raise ValueError(
                            f"MCP server '{server_name}' uses localhost URL '{url}'. "
                            "For type='response', the MCP server must be publicly accessible. "
                            "Please use a public IP address or domain name, and ensure the server "
                            "is bound to '0.0.0.0' (not '127.0.0.1') to accept external connections."
                        )
                    tool_dict["server_url"] = url
                    # Note: OpenAI Responses API doesn't accept a "transport" field
                    # The transport type is inferred from the server_url
                else:
                    raise ValueError(f"MCP server '{server_name}' with transport '{transport}' requires 'url' field")
            elif transport == "stdio":
                raise ValueError(
                    f"MCP server '{server_name}' uses 'stdio' transport, which is not supported by OpenAI Responses API. "
                    "Please use 'streamable_http', 'http', or 'sse' transport, or disable native OpenAI tools "
                    "(code_interpreter, web_search, etc.) to use ChatCompletion API instead."
                )
            else:
                raise ValueError(f"Unsupported MCP transport type: {transport}")
            
            tools.append(tool_dict)
        
        return tools

