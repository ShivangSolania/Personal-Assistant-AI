"""
mcp/tools/calculator.py
───────────────────────
Safe math expression evaluator (MCP-compatible).
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

TOOL_NAME = "calculator"
TOOL_DESCRIPTION = (
    "Evaluate a mathematical expression safely. "
    "Supports +, -, *, /, **, %, and common math functions (sqrt, sin, cos, log, abs)."
)
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "expression": {
            "type": "string",
            "description": "Math expression to evaluate, e.g. '(2 + 3) * 4'",
        },
    },
    "required": ["expression"],
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "expression": {"type": "string"},
        "result": {"type": "number"},
    },
}

# ── Safe evaluator ────────────────────────────────────────

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "abs": abs,
    "ceil": math.ceil,
    "floor": math.floor,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an AST node using whitelisted operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        return _ALLOWED_OPS[op_type](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return _ALLOWED_OPS[op_type](_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS:
            func = _ALLOWED_FUNCS[node.func.id]
            args = [_safe_eval(a) for a in node.args]
            return float(func(*args))
        raise ValueError(f"Function not allowed: {ast.dump(node.func)}")
    if isinstance(node, ast.Name) and node.id in _ALLOWED_FUNCS:
        val = _ALLOWED_FUNCS[node.id]
        if isinstance(val, (int, float)):
            return float(val)
        raise ValueError(f"'{node.id}' is a function, not a constant")
    raise ValueError(f"Unsupported expression node: {ast.dump(node)}")


async def handler(expression: str, **_kwargs: Any) -> dict[str, Any]:
    """Evaluate *expression* safely and return the result."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        return {"expression": expression, "result": result}
    except Exception as exc:
        raise ValueError(f"Cannot evaluate '{expression}': {exc}") from exc


def register(registry) -> None:
    """Auto-discovery hook."""
    registry.register(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=INPUT_SCHEMA,
        output_schema=OUTPUT_SCHEMA,
        handler=handler,
    )
