from langgraph.prebuilt import ToolNode

def multiply(a: float, b: float) -> float:
    """Multiply a and b."""
    return a * b

def add(a: float, b: float) -> float:
    """Add a and b."""
    return a + b

def divide(a: float, b: float) -> float:
    """Divide a by b."""
    return a / b

tools = [multiply, add, divide]

tools_node = ToolNode(tools) 