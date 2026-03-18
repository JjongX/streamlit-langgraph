# MCP Server Examples

This folder contains sample MCP servers used by the Streamlit examples.

## Start the math MCP server

Run in terminal A:

```bash
cd /home/jongs/streamlit-langgraph
source /home/jongs/envs/streamlit-langgraph/bin/activate
fastmcp run examples/mcp_servers/math_server.py --transport streamable-http --host 127.0.0.1 --port 8000
```

The server will be available at:

`http://127.0.0.1:8000/mcp`

## If your MCP server is running elsewhere

Point your app config to that endpoint (example path: `examples/configs/08_mcp_example.yaml`):

```yaml
mcp_servers:
  math:
    transport: streamable_http
    url: http://<host>:<port>/mcp
```
