"""
executor.py
===========
Executes individual plan steps by calling tool functions in tools.py directly,
bypassing the ReAct agent. Each step dict comes from the plan produced by
GSMPlanner and carries a "tool" name and "user_inputs" dict with values
already filled in by the user via the plan UI.
"""

from planner.tool_registry import TOOL_FN_MAP


def _coerce(value):
    if not isinstance(value, str):
        return value
    v = value.strip()
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    if "," in v:
        return [item.strip() for item in v.split(",") if item.strip()]
    return v


def execute_step(step: dict) -> dict:
    """
    Run a single plan step and return the result.

    Builds kwargs from step["user_inputs"] — only entries whose "value"
    is not None are forwarded to the tool function.

    Returns a dict that is safe to JSON-serialize and display in the UI.
    """
    tool = step.get("tool", "manual")

    if tool == "manual":
        return {
            "status": "manual",
            "message": "This step requires manual execution outside the tool.",
        }

    fn = TOOL_FN_MAP.get(tool)
    if fn is None:
        return {"error": f"Unknown tool: '{tool}'. Available: {list(TOOL_FN_MAP)}"}

    kwargs = {
        k: _coerce(v["value"])
        for k, v in step.get("user_inputs", {}).items()
        if isinstance(v, dict) and v.get("value") is not None
    }

    try:
        result = fn(**kwargs)
    except Exception as e:
        return {"error": str(e), "tool": tool, "kwargs": kwargs}

    # Ensure the result is JSON-serialisable (tools return str, dict, or DataFrame)
    if isinstance(result, str):
        return {"output": result}
    if isinstance(result, dict):
        return result
    # Fallback for unexpected types (e.g. pandas DataFrame returned directly)
    try:
        import pandas as pd
        if isinstance(result, pd.DataFrame):
            return {"output": result.to_dict(orient="records")}
    except ImportError:
        pass
    return {"output": str(result)}
