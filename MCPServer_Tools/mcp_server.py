import asyncio
import json
from mcp.server import Server
import mcp.server.stdio
import mcp.types as types

# Initialize the native MCP server
server = Server("Electronic Pricing Server")

# Mock database containing electronic device details
DEVICE_PRICES = {
    "iphone 16 pro": {"price": 999.00, "currency": "USD", "stock": "In Stock", "category": "Smartphone"},
    "samsung galaxy s25": {"price": 899.00, "currency": "USD", "stock": "In Stock", "category": "Smartphone"},
    "macbook pro 16": {"price": 2499.00, "currency": "USD", "stock": "Low Stock", "category": "Laptop"},
    "sony wh-1000xm5": {"price": 399.00, "currency": "USD", "stock": "In Stock", "category": "Headphones"},
    "ipad pro 11": {"price": 799.00, "currency": "USD", "stock": "In Stock", "category": "Tablet"}
}

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """Define the tools exposed to AutoGen."""
    return [
        types.Tool(
            name="get_electronic_price",
            description="Get the price details, stock status, and category for a given electronic device.",
            inputSchema={
                "type": "object",
                "properties": {
                    "device_name": {
                        "type": "string",
                        "description": "The name or model of the electronic device (e.g., 'iPhone 16 Pro', 'MacBook Pro 16')."
                    }
                },
                "required": ["device_name"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool execution requests from AutoGen."""
    if name == "get_electronic_price":
        device_name = arguments.get("device_name", "")
        key = device_name.lower().strip()
        
        if key in DEVICE_PRICES:
            result = DEVICE_PRICES[key]
        else:
            result = {
                "error": f"Device '{device_name}' not found in the pricing database.",
                "available_devices": list(DEVICE_PRICES.keys())
            }
            
        return [types.TextContent(type="text", text=json.dumps(result))]
    
    raise ValueError(f"Unknown tool: {name}")

async def main():
    # Run the server securely over stdio streams without any banner prints
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())