"""
tool_registry.py
================
Central registry for planner-facing tools.

Reads _planner_meta attributes from functions in tools.py and builds:
  TOOL_FN_MAP         — {tool_name: fn} for executor.py
  TOOL_USER_INPUTS    — {tool_name: {param: {description, value, required}}}
                        for planner_core.py validation and UI input collection
  AVAILABLE_TOOLS_TEXT — formatted string injected into the LLM planning prompt

The "manual" pseudo-tool (no real function) is registered separately.

Adding a new tool requires only:
  1. Defining the function in tools.py
  2. Attaching a _planner_meta dict to it
  3. Adding the function to _TOOL_FUNCTIONS below
"""

from tools import (
    load_model,
    run_fba,
    run_pfba,
    run_geometric_fba,
    run_fva,
    gene_knockout_simulation,
    reaction_knockout_simulation,
    sample_metabolic_model,
    set_model_objective,
    visualize_escher,
    list_escher_maps,
    run_memote_report,
    add_reaction,
    set_reaction_bounds,
    build_model_with_carveme,
    build_context_model_with_corda,
    find_essential_genes,
    find_essential_reactions,
    check_model_consistency,
    check_mass_balance,
    prune_unused_reactions,
    prune_unused_metabolites,
    find_minimal_medium,
    build_model_with_mackinac,
    run_moma,
    run_room,
    find_blocked_rxns,
    remove_genes,
    gapfill,
    gapfill_model_with_carveme,
)

# Order controls the sequence in AVAILABLE_TOOLS_TEXT shown to the LLM
_TOOL_FUNCTIONS = [
    load_model,
    run_fba,
    run_pfba,
    run_geometric_fba,
    run_fva,
    gene_knockout_simulation,
    reaction_knockout_simulation,
    sample_metabolic_model,
    set_model_objective,
    visualize_escher,
    list_escher_maps,
    run_memote_report,
    add_reaction,
    set_reaction_bounds,
    build_model_with_carveme,
    build_context_model_with_corda,
    find_essential_genes,
    find_essential_reactions,
    check_model_consistency,
    check_mass_balance,
    prune_unused_reactions,
    prune_unused_metabolites,
    find_minimal_medium,
    build_model_with_mackinac,
    run_moma,
    run_room,
    find_blocked_rxns,
    remove_genes,
    gapfill,
    gapfill_model_with_carveme,
]


def _build_registry(fns: list) -> tuple[dict, dict, dict]:
    fn_map: dict = {}
    user_inputs_map: dict = {}
    desc_map: dict = {}

    for fn in fns:
        meta = getattr(fn, "_planner_meta", None)
        if meta is None:
            raise AttributeError(
                f"Tool function '{fn.__name__}' is missing a _planner_meta attribute. "
                "Add one in tools.py before registering it here."
            )
        name = meta["name"]
        fn_map[name] = fn
        desc_map[name] = meta["description"]
        # _planner_meta uses "default" for the static default value;
        # TOOL_USER_INPUTS uses "value" as the runtime-mutable slot filled by the UI.
        user_inputs_map[name] = {
            param_name: {
                "description": p["description"],
                "value":       p["default"],
                "required":    p["required"],
                # Optional UI hints: "input_type": "file" makes the plan editor
                # render a file uploader instead of a text box; "file_types"
                # restricts the accepted extensions.
                **({"input_type": p["input_type"]} if "input_type" in p else {}),
                **({"file_types": p["file_types"]} if "file_types" in p else {}),
            }
            for param_name, p in meta.get("params", {}).items()
        }

    return fn_map, user_inputs_map, desc_map


_fn_map, _user_inputs, _descs = _build_registry(_TOOL_FUNCTIONS)

# "manual" pseudo-tool: no function, no params
_user_inputs["manual"] = {}
_descs["manual"] = "Step requires manual user action — no automated tool available"

# ── Public exports ─────────────────────────────────────────────────────────────

TOOL_FN_MAP: dict = _fn_map

TOOL_USER_INPUTS: dict = _user_inputs

AVAILABLE_TOOLS_TEXT: str = (
    "Available tools (use these EXACT names in the plan output):\n"
    + "\n".join(
        f"  {name:<32} {desc}"
        for name, desc in _descs.items()
    )
)
