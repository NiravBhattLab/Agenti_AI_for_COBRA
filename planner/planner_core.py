"""
planner_core.py
===============
Workflow planning module for the GSM multi-step planning system.

Takes retrieval results from retrieve.py and produces a structured,
executable plan that can be displayed to the user for approval and
then executed step-by-step through the existing COBRApy agent.

Responsibilities
----------------
1. Fetch full step details from ChromaDB for the top-scored tier 2
   sub-workflows (since the retrieval result only returns top-k steps,
   not necessarily all steps in a selected sub-workflow).
2. Build a structured candidate set for the LLM planner.
3. Make an LLM call to select, order, and explain the plan steps.
4. Parse and validate the LLM output into a typed plan dict.
5. Map technique fields to available COBRApy tools.
6. Support plan revision when the user provides text feedback.

Plan structure
--------------
The output plan is a dict consumed by the FastAPI endpoint and
Streamlit UI:

{
  "query":   str,
  "summary": str,   # 1-2 sentence description of what the plan achieves
  "plan": [
    {
      "step_number":        int,
      "step_name":          str,
      "description":        str,
      "tool":               str,   # exact tool name or "manual"
      "requires_user_input": bool,
      "user_inputs": {
        "param_name": {
          "description": str,   # shown to user in UI input prompt
          "value":       any,   # None = must be filled before execution
          "required":    bool
        }
      },
      "source_paper":   str,   # paper_id the step came from
      "source_step_id": str,   # original step_id (e.g. "S1")
      "included":       bool,  # user can toggle this in the UI
      "rationale":      str    # why this step is in the plan
    }
  ]
}

LLM callable interface
-----------------------
The planner accepts any callable matching:
  llm_fn(prompt: str, system_prompt: str) -> str

This keeps the planner decoupled from the LLM provider. In this app,
main.py provides a bridge function that wraps the existing Llama-Index
LLM instance to match this interface.

Usage
-----
  from planner.retrieve import GSMRetriever
  from planner.planner_core import GSMPlanner

  retriever = GSMRetriever()
  planner   = GSMPlanner(llm_fn=my_llm_fn, collection=retriever.collection)

  result    = retriever.retrieve("simulate SDH inhibition and run FVA")
  plan      = planner.plan("simulate SDH inhibition and run FVA", result)

  # After user edits:
  revised   = planner.revise_plan(
      query="simulate SDH inhibition and run FVA",
      current_plan=plan,
      user_feedback="Also add a gene knockout simulation after FVA"
  )

Dependencies
------------
  pip install chromadb
"""

import copy
import json
import re
from typing import Callable, Optional

import chromadb

from planner.tool_registry import TOOL_USER_INPUTS, AVAILABLE_TOOLS_TEXT

# ── Technique alias fallback ──────────────────────────────────────────────────
# Only retains genuinely opaque aliases the LLM cannot match from descriptions.
# Common names (FBA, FVA, Escher, COBRApy) are dropped — the LLM reasons from
# AVAILABLE_TOOLS_TEXT descriptions directly.

TECHNIQUE_TO_TOOL: dict[str, str] = {
    "fluxVariability":       "run_fva",
    "flux_variability":      "run_fva",
    "ACHR":                  "sample_metabolic_model",
    "OptGP":                 "sample_metabolic_model",
    "single_gene_deletion":  "gene_knockout_simulation",
    "double_gene_deletion":  "gene_knockout_simulation",
    "pFBA":                  "run_pfba",
    "escher":                "visualize_escher",
    "escher_visualization":  "visualize_escher",
}

# ── System prompts ────────────────────────────────────────────────────────────

_PLANNING_SYSTEM_PROMPT = """You are a workflow planning assistant for constraint-based genome-scale metabolic modeling using COBRApy and the COBRA toolbox.

Your task is to build a step-by-step executable plan from candidate workflow steps retrieved from published metabolic modeling papers. The plan will be reviewed by the user and then executed using the available COBRApy tools.

Rules:
- Select ONLY from the candidate steps provided — do not invent steps not in the candidates
- Model loading (load_model) must always be the first step if present
- Order steps in a logical execution sequence
- Use EXACT tool names from the available tools list
- For each step, specify what user inputs (if any) are needed before execution
- Write a brief, informative rationale for each included step
- Output ONLY valid JSON matching the schema provided — no preamble, no markdown fences
"""

_REVISION_SYSTEM_PROMPT = """You are a workflow planning assistant for constraint-based genome-scale metabolic modeling.

Your task is to revise an existing executable plan based on user feedback. Make only the changes the feedback requests. Preserve the structure and rationale of unchanged steps. Output ONLY valid JSON — no preamble, no markdown fences.
"""

# ── Output schema shown to the LLM ───────────────────────────────────────────

_PLAN_SCHEMA = """
{
  "summary": "1-2 sentence description of what this plan achieves overall.",
  "plan": [
    {
      "step_number": 1,
      "step_name": "Short descriptive name for this step",
      "description": "What this step does and why it is needed in this workflow.",
      "tool": "exact_tool_name_or_manual",
      "requires_user_input": true,
      "user_inputs": {
        "param_name": {
          "description": "What this parameter is and example values",
          "value": null,
          "required": true
        }
      },
      "source_paper": "paper_id string",
      "source_step_id": "S1",
      "included": true,
      "rationale": "Why this step is included in the plan for this specific query."
    }
  ]
}
"""


# ── Local llama.cpp LLM factory ───────────────────────────────────────────────

def make_local_llm_fn(
    model_path:   str,
    n_ctx:        int = 8192,
    n_gpu_layers: int = -1,
    max_tokens:   int = 2048,
) -> "Callable[[str, str], str]":
    """
    Load a local GGUF model via llama.cpp and return an llm_fn callable
    compatible with GSMPlanner(llm_fn=...).

    The model is loaded once; the returned function is called once per
    LLM request (plan generation or revision).

    Thinking blocks from Qwen3 (<think>...</think>) and Gemma4
    (<|channel>thought...<channel|>) are stripped automatically before
    the response is returned.

    Parameters
    ----------
    model_path : str
        Absolute path to the .gguf model file.
        Defaults: QWEN_MODEL_PATH or GEMMA_MODEL_PATH env vars (see CLI).
    n_ctx : int
        Context window size. 8192 is safe for planner prompts.
    n_gpu_layers : int
        Layers to offload to GPU. -1 = all layers.
    max_tokens : int
        Maximum output tokens per call. 2048 is sufficient for a plan JSON.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama_cpp is not installed. Run: pip install llama-cpp-python"
        )

    print(f"Loading local model : {model_path}")
    print(f"  n_ctx={n_ctx}  n_gpu_layers={n_gpu_layers}  max_tokens={max_tokens}")
    _llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        verbose=False,
    )
    print("  Model loaded.\n")

    def _call(prompt: str, system_prompt: str) -> str:
        response = _llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        raw = response["choices"][0]["message"]["content"].strip()
        # Strip thinking blocks (Qwen3 <think>...</think> / Gemma4 <|channel>thought...)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"<\|channel>thought\n.*?<channel\|>", "", raw, flags=re.DOTALL).strip()
        return raw

    return _call


# ── Planner class ─────────────────────────────────────────────────────────────

class GSMPlanner:
    """
    Builds and revises executable workflow plans from retrieval results.

    Parameters
    ----------
    llm_fn : Callable[[str, str], str]
        Function that takes (prompt, system_prompt) and returns the LLM
        response string. Decouples the planner from the LLM provider.
        In this app, main.py provides a bridge that wraps the Llama-Index
        LLM instance to match this interface.

    collection : chromadb.Collection
        The ChromaDB collection. Used for targeted step lookups beyond
        what the retrieval phase returned.

    top_subworkflows : int
        How many of the top-scored tier 2 hits to consider as plan seeds.
        Default 3. The LLM selects the most relevant from these.

    max_retries : int
        How many times to retry the LLM call if JSON parsing fails.
    """

    def __init__(
        self,
        llm_fn:           Callable[[str, str], str],
        collection:       chromadb.Collection,
        top_subworkflows: int = 3,
        max_retries:      int = 3,
    ):
        self.llm_fn           = llm_fn
        self.collection       = collection
        self.top_subworkflows = top_subworkflows
        self.max_retries      = max_retries

    # ── Public interface ──────────────────────────────────────────────────────

    def plan(self, query: str, retrieval_result: dict) -> dict:
        """
        Generate an initial workflow plan from a retrieval result.

        Parameters
        ----------
        query : str
            The original user query.
        retrieval_result : dict
            Output of GSMRetriever.retrieve().

        Returns
        -------
        dict
            A plan dict with keys: query, summary, plan (list of step dicts).
        """
        candidates = self._build_candidates(retrieval_result)
        prompt     = self._build_planning_prompt(query, candidates)
        raw        = self._call_llm_with_retry(prompt, _PLANNING_SYSTEM_PROMPT)
        plan       = self._parse_and_validate(raw, query)
        return plan

    def revise_plan(
        self,
        query:         str,
        current_plan:  dict,
        user_feedback: str,
    ) -> dict:
        """
        Revise an existing plan based on free-text user feedback.

        Parameters
        ----------
        query : str
            The original user query (for context).
        current_plan : dict
            The current plan dict (output of plan() or a previous revise_plan()).
        user_feedback : str
            The user's natural language feedback, e.g.
            "Also add a gene knockout after FVA"
            "Remove step 3 — I don't need the glucose deprivation scenario"
            "Why is step 2 included?"

        Returns
        -------
        dict
            Revised plan dict in the same format as plan().
        """
        prompt = self._build_revision_prompt(query, current_plan, user_feedback)
        raw    = self._call_llm_with_retry(prompt, _REVISION_SYSTEM_PROMPT)
        plan   = self._parse_and_validate(raw, query)
        return plan

    # ── Candidate building ────────────────────────────────────────────────────

    def _build_candidates(self, retrieval_result: dict) -> dict:
        """
        Organize retrieval results into a structured candidate set for the LLM.

        For each top-scored tier 2 hit, fetches the full tier 3 step details
        for all steps in the group — not just the ones that happened to surface
        in the top-k tier 3 hits.

        Returns
        -------
        dict with keys:
          subworkflows : list of sub-workflow candidates (from tier 2 hits)
          supplementary: list of individual step candidates not covered by
                         any selected sub-workflow (from tier 3 hits)
        """
        tier2_hits = retrieval_result["phase2"]["tier2_hits"]
        tier3_hits = retrieval_result["phase2"]["tier3_hits"]

        # Take top N tier 2 hits as primary plan seeds
        top_t2 = tier2_hits[: self.top_subworkflows]

        # Track which (paper_id, step_id) pairs are covered by tier 2 groups
        covered: set[tuple[str, str]] = set()

        subworkflows = []
        for hit in top_t2:
            paper_id = hit["paper_id"]
            step_ids = hit["step_ids"]

            # Fetch detailed tier 3 data for each step in this group
            step_details = self._fetch_step_details(paper_id, step_ids)

            subworkflows.append({
                "score":        hit["score"],
                "paper_id":     paper_id,
                "entry_id":     hit["entry_id"],
                "step_ids":     step_ids,
                "objective":    hit["document"],
                "step_category":hit["step_category"],
                "steps":        step_details,
            })

            for sid in step_ids:
                covered.add((paper_id, sid))

        # Supplementary: tier 3 hits not already covered by any tier 2 group
        supplementary = []
        for hit in tier3_hits:
            paper_id = hit["paper_id"]
            step_ids = hit["step_ids"]
            step_id  = step_ids[0] if step_ids else ""

            if (paper_id, step_id) not in covered:
                technique = hit.get("technique", "")
                tool, requires_input, inputs = self._map_tool(technique)
                supplementary.append({
                    "score":         hit["score"],
                    "paper_id":      paper_id,
                    "step_id":       step_id,
                    "step_category": hit["step_category"],
                    "technique":     technique,
                    "tool":          tool,
                    "requires_user_input": requires_input,
                    "user_inputs":   inputs,
                    "description":   hit["document"],
                })
                covered.add((paper_id, step_id))

        return {"subworkflows": subworkflows, "supplementary": supplementary}

    def _fetch_step_details(
        self,
        paper_id: str,
        step_ids: list[str],
    ) -> list[dict]:
        """
        Fetch tier 3 entries from ChromaDB for specific step_ids in a paper.
        Returns a list of step detail dicts ordered by original step_id.
        """
        if not step_ids:
            return []

        entry_ids = [f"T3_{sid}" for sid in step_ids]

        try:
            results = self.collection.get(
                where={"$and": [
                    {"paper_id": paper_id},
                    {"tier":     3},
                    {"entry_id": {"$in": entry_ids}},
                ]},
                include=["documents", "metadatas"],
            )
        except Exception as e:
            print(f"  Step detail lookup failed for {paper_id}: {e}")
            return []

        # Map entry_id → result for ordered reconstruction
        by_entry: dict[str, dict] = {}
        for doc, meta in zip(results["documents"], results["metadatas"]):
            technique = meta.get("technique", "")
            tool, requires_input, inputs = self._map_tool(technique)
            by_entry[meta["entry_id"]] = {
                "step_id":           json.loads(meta.get("step_ids", "[]"))[0] if meta.get("step_ids") else "",
                "step_category":     meta.get("step_category", ""),
                "technique":         technique,
                "tool":              tool,
                "requires_user_input": requires_input,
                "user_inputs":       inputs,
                "description":       doc,
            }

        # Return in original step_id order
        ordered = []
        for eid in entry_ids:
            if eid in by_entry:
                ordered.append(by_entry[eid])

        return ordered

    # ── Tool mapping ──────────────────────────────────────────────────────────

    def _map_tool(self, technique: str) -> tuple[str, bool, dict]:
        """
        Map a technique string to a tool name, requires_user_input flag,
        and default user_inputs template.

        Returns
        -------
        (tool_name, requires_user_input, user_inputs_dict)
        """
        tool = TECHNIQUE_TO_TOOL.get(technique, "manual")
        inputs = copy.deepcopy(TOOL_USER_INPUTS.get(tool, {}))
        requires_input = bool(inputs) and any(
            v.get("required", False) and v.get("value") is None
            for v in inputs.values()
        )
        return tool, requires_input, inputs

    # ── Prompt builders ───────────────────────────────────────────────────────

    def _build_planning_prompt(self, query: str, candidates: dict) -> str:
        subworkflows  = candidates["subworkflows"]
        supplementary = candidates["supplementary"]

        # Format sub-workflows
        sw_block = ""
        for i, sw in enumerate(subworkflows, 1):
            sw_block += (
                f"\nSub-workflow {i} (similarity score: {sw['score']:.3f})\n"
                f"  Paper    : {sw['paper_id']}\n"
                f"  Category : {sw['step_category']}\n"
                f"  Steps    : {sw['step_ids']}\n"
                f"  Overview : {sw['objective'][:300]}{'...' if len(sw['objective']) > 300 else ''}\n"
            )
            if sw["steps"]:
                sw_block += "  Step details:\n"
                for s in sw["steps"]:
                    sw_block += (
                        f"    [{s['step_id']}] tool={s['tool']} | technique={s['technique']}\n"
                        f"          {s['description'][:200]}{'...' if len(s['description']) > 200 else ''}\n"
                    )

        # Format supplementary steps
        supp_block = ""
        if supplementary:
            for s in supplementary:
                supp_block += (
                    f"\n  [{s['step_id']}] score={s['score']:.3f} | "
                    f"tool={s['tool']} | technique={s['technique']}\n"
                    f"        {s['description'][:200]}{'...' if len(s['description']) > 200 else ''}\n"
                )
        else:
            supp_block = "\n  (none)\n"

        return f"""USER QUERY:
{query}

CANDIDATE SUB-WORKFLOWS (retrieved from published papers — primary plan sources):
{sw_block}

SUPPLEMENTARY INDIVIDUAL STEPS (not covered by any sub-workflow above):
{supp_block}

{AVAILABLE_TOOLS_TEXT}

INSTRUCTIONS:
1. Select the sub-workflow(s) most relevant to the user's query as your primary plan basis
2. Add supplementary steps only if they are genuinely needed and not already covered
3. Remove steps that are not relevant to what the user is asking for
4. Ensure model loading (load_model) is always step 1 if present in candidates
5. Order all remaining steps in a logical execution sequence
6. For each step, specify the exact tool name and any user inputs needed
7. If a step has technique "Manual curation", use tool "manual"
8. Write a clear rationale for why each step is included

SEQUENCE INSTRUCTIONS:
1. Everytime there is a step performing an optimization step, there needs to be an escher visualization step immediately after

OUTPUT SCHEMA (return ONLY valid JSON, no preamble):
{_PLAN_SCHEMA}
"""

    def _build_revision_prompt(
        self,
        query:        str,
        current_plan: dict,
        feedback:     str,
    ) -> str:
        plan_text = json.dumps(current_plan, indent=2)

        return f"""ORIGINAL QUERY:
{query}

CURRENT PLAN:
{plan_text}

USER FEEDBACK:
{feedback}

INSTRUCTIONS:
- Make only the changes the feedback requests
- If the feedback asks to add a step, insert it in a logical position and renumber
- If the feedback asks to remove a step, set "included": false (do not delete it)
- If the feedback asks to explain a step, update the rationale field for that step
- Preserve the structure and content of all other steps unchanged
- Update the summary if the plan has meaningfully changed

OUTPUT SCHEMA (return ONLY valid JSON, no preamble):
{_PLAN_SCHEMA}
"""

    # ── LLM call and parsing ──────────────────────────────────────────────────

    def _call_llm_with_retry(self, prompt: str, system_prompt: str) -> str:
        """Call the LLM up to max_retries times, returning the raw string."""
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.llm_fn(prompt, system_prompt)
            except Exception as e:
                print(f"  [attempt {attempt}/{self.max_retries}] LLM call failed: {e}")
                if attempt == self.max_retries:
                    raise
        return ""

    def _parse_and_validate(self, raw: str, query: str) -> dict:
        """
        Parse the LLM's JSON output and validate/repair the plan structure.

        Handles:
          - Markdown code fence stripping
          - Missing or malformed fields (fills in safe defaults)
          - Tool name normalization
          - Step number re-sequencing
          - User_inputs template injection for known tools
        """
        # Strip markdown fences if present
        text = raw.strip()
        if "```" in text:
            # Extract the JSON block between the first { and last }
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start != -1 and end > start:
                text = text[start:end]

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"  JSON parse failed: {e}. Returning empty plan.")
            return self._empty_plan(query)

        steps = data.get("plan", [])
        if not isinstance(steps, list):
            return self._empty_plan(query)

        validated_steps = []
        for i, step in enumerate(steps, 1):
            if not isinstance(step, dict):
                continue

            # Normalize tool name
            tool = step.get("tool", "manual")
            if tool not in TOOL_USER_INPUTS:
                # Try to recover by checking technique
                technique = step.get("technique", "")
                tool = TECHNIQUE_TO_TOOL.get(technique, "manual")

            # Normalize user_inputs to canonical template keys for this tool.
            # The LLM may use CobraPy parameter names (e.g. 'gene_list') that
            # differ from our function signatures ('gene_names'). We rebuild
            # from the template and carry over values: exact key match first,
            # then pair any leftover LLM values with still-unfilled required keys.
            user_inputs = step.get("user_inputs", {})
            template = TOOL_USER_INPUTS.get(tool, {})
            if template:
                canonical = copy.deepcopy(template)
                unmatched_llm: list = []
                for llm_key, llm_val in user_inputs.items():
                    val = llm_val["value"] if isinstance(llm_val, dict) and "value" in llm_val else llm_val
                    if llm_key in canonical:
                        canonical[llm_key]["value"] = val
                    else:
                        unmatched_llm.append(val)
                unfilled = [k for k in canonical if canonical[k].get("required") and canonical[k]["value"] is None]
                for val, tmpl_key in zip(unmatched_llm, unfilled):
                    canonical[tmpl_key]["value"] = val
                user_inputs = canonical

            requires_input = bool(user_inputs) and any(
                v.get("required", False) and v.get("value") is None
                for v in user_inputs.values()
                if isinstance(v, dict)
            )

            validated_steps.append({
                "step_number":         i,
                "step_name":           step.get("step_name", f"Step {i}"),
                "description":         step.get("description", ""),
                "tool":                tool,
                "requires_user_input": requires_input,
                "user_inputs":         user_inputs,
                "source_paper":        step.get("source_paper", ""),
                "source_step_id":      step.get("source_step_id", ""),
                "included":            step.get("included", True),
                "rationale":           step.get("rationale", ""),
            })

        return {
            "query":   query,
            "summary": data.get("summary", ""),
            "plan":    validated_steps,
        }

    def _empty_plan(self, query: str) -> dict:
        return {"query": query, "summary": "Plan generation failed.", "plan": []}


# ── Print helper ──────────────────────────────────────────────────────────────

def print_plan(plan: dict) -> None:
    """Pretty-print a plan dict for debugging and inspection."""
    print(f"\n{'═' * 70}")
    print(f"  Query   : {plan['query']}")
    print(f"  Summary : {plan['summary']}")
    print(f"{'═' * 70}\n")

    for step in plan["plan"]:
        status = "✓" if step["included"] else "✗"
        inputs = " [needs input]" if step["requires_user_input"] else ""
        print(f"  [{status}] Step {step['step_number']}: {step['step_name']}")
        print(f"       Tool      : {step['tool']}{inputs}")
        print(f"       Source    : {step['source_paper']} / {step['source_step_id']}")
        print(f"       Rationale : {step['rationale']}")
        if step["user_inputs"]:
            print(f"       Inputs:")
            for k, v in step["user_inputs"].items():
                val = v.get("value", None)
                req = " (required)" if v.get("required") else " (optional)"
                print(f"         {k}: {val}{req} — {v.get('description','')}")
        print()


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from planner.retrieve import GSMRetriever

    parser = argparse.ArgumentParser(description="Test the planner module.")
    parser.add_argument("--query",        required=True,  help="User query string")
    parser.add_argument("--chroma-dir",   default=None)
    parser.add_argument("--collection",   default=None)
    parser.add_argument("--llm",          default="local",
                        help="LLM mode: 'local' (default) uses local llama.cpp model, "
                             "'print' shows prompt without calling LLM, "
                             "'openai' uses OpenAI gpt-4o (requires OPENAI_API_KEY)")
    # Local model flags (used when --llm local)
    parser.add_argument("--model-path",   default=None,
                        help="Path to GGUF model file. Falls back to QWEN_MODEL_PATH "
                             "or GEMMA_MODEL_PATH env var if not set.")
    parser.add_argument("--n-ctx",        type=int, default=8192,
                        help="Context window size (default: 8192)")
    parser.add_argument("--n-gpu-layers", type=int, default=-1,
                        help="GPU layers to offload (-1=all, default: -1)")
    parser.add_argument("--max-tokens",   type=int, default=2048,
                        help="Max output tokens per LLM call (default: 2048)")
    args = parser.parse_args()

    # Build the LLM callable based on --llm flag
    if args.llm == "local":
        import os
        model_path = (
            args.model_path
            or os.environ.get("QWEN_MODEL_PATH")
            or os.environ.get("GEMMA_MODEL_PATH")
        )
        if not model_path:
            raise ValueError(
                "For --llm local, provide --model-path or set the "
                "QWEN_MODEL_PATH (or GEMMA_MODEL_PATH) environment variable."
            )
        llm_fn = make_local_llm_fn(
            model_path=model_path,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            max_tokens=args.max_tokens,
        )

    elif args.llm == "print":
        def llm_fn(prompt, system_prompt):
            print("\n" + "─" * 70)
            print("SYSTEM PROMPT:\n" + system_prompt)
            print("─" * 70)
            print("PLANNING PROMPT (first 2000 chars):\n" + prompt[:2000])
            print("─" * 70)
            return '{"summary": "[dry run — no LLM called]", "plan": []}'

    elif args.llm == "openai":
        import os
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        def llm_fn(prompt, system_prompt):
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system",  "content": system_prompt},
                    {"role": "user",    "content": prompt},
                ],
                temperature=0.0,
            )
            return resp.choices[0].message.content

    else:
        raise ValueError(f"Unknown --llm option: {args.llm}")

    # Build kwargs for retriever
    retriever_kwargs = {}
    if args.chroma_dir:
        retriever_kwargs["chroma_dir"] = args.chroma_dir
    if args.collection:
        retriever_kwargs["collection_name"] = args.collection

    retriever = GSMRetriever(**retriever_kwargs)

    result = retriever.retrieve(args.query)
    print(f"\nRetrieval complete:")
    print(f"  Tier 2 hits: {len(result['phase2']['tier2_hits'])}")
    print(f"  Tier 3 hits: {len(result['phase2']['tier3_hits'])}")

    planner = GSMPlanner(
        llm_fn=llm_fn,
        collection=retriever.collection,
    )

    plan = planner.plan(args.query, result)
    print_plan(plan)
