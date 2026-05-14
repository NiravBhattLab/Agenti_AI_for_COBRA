"""
automata_ground_truth.py
========================
Ground-truth reference script for validating the Automata metabolic
modeling agent against direct COBRApy outputs.

Run this BEFORE testing the agent. The values printed here are what the
agent should match (within tolerance). Each section maps to a Scientific
Accuracy test in the capability test plan.

Usage:
    pip install cobra
    python automata_ground_truth.py [--section SECTION] [--json]

    --section   Run only one section (e.g. --section fba)
                Options: fba, fva, gene_ko, rxn_ko, sampling, objectives,
                         inspection, bounds
    --json      Output all results as a single JSON blob to stdout
                (useful for piping into a diff tool)
"""

import argparse
import json
import sys
import warnings
from io import StringIO

warnings.filterwarnings("ignore")  # suppress solver verbosity

# ── dependency check ───────────────────────────────────────────────────────────
try:
    import cobra
    from cobra.io import load_model, read_sbml_model
    from cobra.flux_analysis import (
        flux_variability_analysis,
        single_gene_deletion,
        double_gene_deletion,
        single_reaction_deletion,
        double_reaction_deletion,
    )
    from cobra.sampling import sample
    import numpy as np
    import pandas as pd
except ImportError as e:
    sys.exit(f"[ERROR] Missing dependency: {e}\n  Run: pip install cobra numpy pandas")

# ── shared constants ───────────────────────────────────────────────────────────
TOLERANCE = 1e-4   # absolute tolerance for comparing floats
MODEL_ID  = "e_coli_core"   # BiGG / COBRApy textbook model
RESULTS   = {}     # global accumulator for --json output

# ── model loader (offline-safe) ────────────────────────────────────────────────
import cobra.data as _cobra_data
import gzip as _gzip
import os as _os
import tempfile as _tempfile
import shutil as _shutil

_TEXTBOOK_GZ = _os.path.join(_os.path.dirname(_cobra_data.__file__), "textbook.xml.gz")

# ── helpers ───────────────────────────────────────────────────────────────────

JSON_MODE = False   # set True by --json flag; redirects all prints to stderr

def _print(*args, **kwargs):
    """Route all output to stderr in JSON mode so stdout stays clean JSON."""
    if JSON_MODE:
        kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


def section(title: str):
    _print("\n" + "═" * 60)
    _print(f"  {title}")
    _print("═" * 60)

def note(text: str):
    _print(f"  ▸ {text}")

def result(key: str, value, description: str = ""):
    tag = f"  ✔  {key}"
    if description:
        tag += f"  ({description})"
    _print(f"{tag}: {value}")
    RESULTS[key] = value

def passed(condition: bool, label: str):
    icon = "  ✔" if condition else "  ✘ FAIL"
    _print(f"{icon}  {label}")
    return condition

def load_fresh():
    """Return a clean copy of the textbook ecoli_core model (offline-safe).

    Tries load_model() first (works if BiGG is reachable), then falls back
    to the .xml.gz bundled with the installed COBRApy package.
    """
    try:
        return load_model(MODEL_ID)
    except Exception:
        pass  # network unavailable — fall through to bundled file
    with _tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        with _gzip.open(_TEXTBOOK_GZ, "rb") as gz:
            _shutil.copyfileobj(gz, tmp)
        tmp_path = tmp.name
    try:
        return read_sbml_model(tmp_path)
    finally:
        _os.unlink(tmp_path)

# ══════════════════════════════════════════════════════════════════════════════
# 1. FLUX BALANCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_fba():
    section("1. FLUX BALANCE ANALYSIS")
    m = load_fresh()

    # 1a. Baseline FBA ─────────────────────────────────────────────────────────
    note("1a. Baseline FBA (ecoli_core default bounds)")
    sol = m.optimize()
    result("fba.baseline.status",    sol.status)
    result("fba.baseline.objective", round(sol.objective_value, 6),
           "mmol/gDW/h biomass")
    result("fba.baseline.objective_rxn",
           list(m.objective.to_json()["expression"]["args"][0]
                .get("args", [{}])[0].get("args", [[""]])[0]
                if "args" in m.objective.to_json()["expression"] else {}),
           "objective reaction ID(s)")

    # Just grab the reaction with highest flux as a sanity reference
    top_rxn = max(sol.fluxes.items(), key=lambda x: abs(x[1]))
    result("fba.baseline.highest_abs_flux_rxn",
           top_rxn[0], f"flux = {round(top_rxn[1], 4)}")

    # 1b. Glucose-restricted FBA ───────────────────────────────────────────────
    note("\n1b. FBA with restricted glucose uptake (EX_glc__D_e lb = -5)")
    m2 = load_fresh()
    m2.reactions.get_by_id("EX_glc__D_e").lower_bound = -5.0
    sol2 = m2.optimize()
    result("fba.glc_restricted.status",    sol2.status)
    result("fba.glc_restricted.objective", round(sol2.objective_value, 6))
    passed(sol2.objective_value < sol.objective_value,
           "Restricted growth < baseline growth")

    # 1c. Infeasible / zero-growth case ────────────────────────────────────────
    note("\n1c. FBA with all carbon exchange reactions blocked")
    m3 = load_fresh()
    for rxn in m3.exchanges:
        rxn.lower_bound = 0  # block all uptake
    sol3 = m3.optimize()
    result("fba.infeasible.status",    sol3.status)
    result("fba.infeasible.objective", round(sol3.objective_value, 6)
           if sol3.status == "optimal" else "N/A")
    note("Agent should return 0 or infeasible — NOT a positive growth rate.")

    # 1d. ATPM flux at baseline ────────────────────────────────────────────────
    note("\n1d. ATPM flux at baseline FBA (reference for objective-switch tests)")
    m4 = load_fresh()
    sol4 = m4.optimize()
    atpm_flux = sol4.fluxes.get("ATPM", None)
    result("fba.baseline.ATPM_flux", round(atpm_flux, 4) if atpm_flux is not None else "N/A")

# ══════════════════════════════════════════════════════════════════════════════
# 2. FLUX VARIABILITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_fva():
    section("2. FLUX VARIABILITY ANALYSIS")
    m = load_fresh()

    target_rxns = ["PFK", "PGI", "ENO", "PGK", "PYK", "FBA"]

    # 2a. FVA at 100% optimum ─────────────────────────────────────────────────
    note("2a. FVA at 100% of optimum (fraction_of_optimum=1.0)")
    fva_100 = flux_variability_analysis(
        m, reaction_list=target_rxns, fraction_of_optimum=1.0
    )
    result("fva.100pct.reactions", target_rxns)
    for rxn_id, row in fva_100.iterrows():
        result(
            f"fva.100pct.{rxn_id}",
            {"min": round(row["minimum"], 6), "max": round(row["maximum"], 6)},
        )

    # 2b. FVA at 90% optimum ──────────────────────────────────────────────────
    note("\n2b. FVA at 90% of optimum (fraction_of_optimum=0.9)")
    fva_90 = flux_variability_analysis(
        m, reaction_list=target_rxns, fraction_of_optimum=0.9
    )
    for rxn_id, row in fva_90.iterrows():
        result(
            f"fva.90pct.{rxn_id}",
            {"min": round(row["minimum"], 6), "max": round(row["maximum"], 6)},
        )
    note("90% ranges should be equal or wider than 100% ranges for each reaction.")

    # 2c. Sanity check: 90% ranges >= 100% ranges ─────────────────────────────
    all_wider = True
    for rxn_id in target_rxns:
        r100 = fva_100.loc[rxn_id]
        r90  = fva_90.loc[rxn_id]
        wider = (r90["minimum"] <= r100["minimum"] + TOLERANCE and
                 r90["maximum"] >= r100["maximum"] - TOLERANCE)
        all_wider = all_wider and wider
    passed(all_wider, "90% FVA ranges ⊇ 100% FVA ranges for all target reactions")

# ══════════════════════════════════════════════════════════════════════════════
# 3. GENE KNOCKOUT SIMULATIONS
# ══════════════════════════════════════════════════════════════════════════════

def run_gene_ko():
    section("3. GENE KNOCKOUT SIMULATIONS")
    m = load_fresh()
    wt_growth = m.optimize().objective_value

    # Test genes selected for specific phenotypes
    gene_cases = {
        "b2779": "essential       (expected ~0 growth)",
        "b4025": "partially impaired  (expected ~0.863)",
        "b0114": "mildly impaired  (expected ~0.797)",
        "b0351": "non-essential   (expected = WT)",
        "b0008": "non-essential   (expected = WT)",
        "b1241": "non-essential   (expected = WT)",
    }

    # 3a. Single gene deletions ────────────────────────────────────────────────
    note("3a. Single gene deletions")
    result("gene_ko.wild_type_growth", round(wt_growth, 6))
    sg_results = single_gene_deletion(m, gene_list=list(gene_cases.keys()))

    for _, row in sg_results.iterrows():
        gene_id = list(row["ids"])[0]
        growth  = row["growth"] if row["status"] == "optimal" else 0.0
        label   = gene_cases.get(gene_id, "")
        result(
            f"gene_ko.single.{gene_id}",
            {"growth": round(float(growth), 6), "status": row["status"]},
            label,
        )

    # Validate categories
    sg_dict = {list(r["ids"])[0]: float(r["growth"]) for _, r in sg_results.iterrows()}
    passed(sg_dict["b2779"] < TOLERANCE,
           "b2779 is essential (growth ≈ 0)")
    passed(abs(sg_dict["b0351"] - wt_growth) < TOLERANCE,
           "b0351 is non-essential (growth = WT)")
    passed(sg_dict["b4025"] < wt_growth - TOLERANCE,
           "b4025 reduces growth below WT")

    # 3b. Double gene deletion ────────────────────────────────────────────────
    note("\n3b. Double gene deletion: b0008 × b0114")
    dg_results = double_gene_deletion(m, gene_list1=["b0008"], gene_list2=["b0114"])
    for _, row in dg_results.iterrows():
        ids_str = "_".join(sorted(row["ids"]))
        result(
            f"gene_ko.double.{ids_str}",
            {"growth": round(float(row["growth"]), 6), "status": row["status"]},
        )

    # 3c. Context-manager single KO (as Automata's tool likely uses) ──────────
    note("\n3c. Context-manager KO for b4025 (verify tool-level approach)")
    with m:
        m.genes.get_by_id("b4025").knock_out()
        sol_ctx = m.optimize()
        result("gene_ko.context_b4025.growth",
               round(sol_ctx.objective_value, 6), "should match single_gene_deletion above")

# ══════════════════════════════════════════════════════════════════════════════
# 4. REACTION KNOCKOUT SIMULATIONS
# ══════════════════════════════════════════════════════════════════════════════

def run_rxn_ko():
    section("4. REACTION KNOCKOUT SIMULATIONS")
    m = load_fresh()
    wt_growth = m.optimize().objective_value

    rxn_cases = {
        "PGI":  "glycolysis — partially dispensable in aerobic conditions",
        "PFK":  "glycolysis — essential in some contexts",
        "ATPM": "ATP maintenance — non-growth-related demand",
        "FBA":  "fructose-bisphosphate aldolase",
        "PGK":  "phosphoglycerate kinase",
    }

    # 4a. Single reaction deletions ───────────────────────────────────────────
    note("4a. Single reaction deletions")
    result("rxn_ko.wild_type_growth", round(wt_growth, 6))
    sr_results = single_reaction_deletion(m, reaction_list=list(rxn_cases.keys()))

    for _, row in sr_results.iterrows():
        rxn_id = list(row["ids"])[0]
        growth = row["growth"] if row["status"] == "optimal" else 0.0
        result(
            f"rxn_ko.single.{rxn_id}",
            {"growth": round(float(growth), 6), "status": row["status"]},
            rxn_cases.get(rxn_id, ""),
        )

    # 4b. Double reaction deletion ────────────────────────────────────────────
    note("\n4b. Double reaction deletion: PGI × FBA")
    dr_results = double_reaction_deletion(m, reaction_list1=["PGI"], reaction_list2=["FBA"])
    for _, row in dr_results.iterrows():
        ids_str = "_".join(sorted(row["ids"]))
        result(
            f"rxn_ko.double.{ids_str}",
            {"growth": round(float(row["growth"]), 6), "status": row["status"]},
        )

    # 4c. Context-manager KO (verify tool approach) ───────────────────────────
    note("\n4c. Context-manager KO for PGI")
    with m:
        m.reactions.get_by_id("PGI").knock_out()
        sol_ctx = m.optimize()
        result("rxn_ko.context_PGI.growth",
               round(sol_ctx.objective_value, 6),
               "should match single_reaction_deletion above")

# ══════════════════════════════════════════════════════════════════════════════
# 5. FLUX SAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def run_sampling():
    section("5. FLUX SAMPLING")
    m = load_fresh()
    N = 500  # keep fast for reference; agent will use user-supplied N

    note(f"Running flux sampling with {N} samples (optgp algorithm)")
    note("Sampling is stochastic — we record mean/std for comparison, not exact values.")

    try:
        s = sample(m, N, method="optgp")
        summary_rxns = ["PFK", "PGI", "ENO", "PYK", "ATPM", "EX_glc__D_e"]
        stats = {}
        for rxn in summary_rxns:
            if rxn in s.columns:
                col = s[rxn]
                stats[rxn] = {
                    "mean":  round(float(col.mean()), 4),
                    "std":   round(float(col.std()),  4),
                    "min":   round(float(col.min()),  4),
                    "max":   round(float(col.max()),  4),
                }
        result("sampling.n_samples",     N)
        result("sampling.n_reactions",   len(s.columns))
        result("sampling.reaction_stats", stats)
        note("When comparing agent output: mean should be within ~2 std of these values.")
        note("Agent output for EX_glc__D_e should be negative (uptake).")
        passed(stats.get("EX_glc__D_e", {}).get("mean", 0) < 0,
               "Mean EX_glc__D_e flux is negative (glucose uptake)")
    except Exception as e:
        note(f"optgp sampling failed ({e}), trying achr...")
        s = sample(m, N, method="achr")
        result("sampling.method_used", "achr (fallback)")
        result("sampling.n_samples",   N)
        result("sampling.n_reactions", len(s.columns))

# ══════════════════════════════════════════════════════════════════════════════
# 6. OBJECTIVE FUNCTION MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def run_objectives():
    section("6. OBJECTIVE FUNCTION MANAGEMENT")
    m = load_fresh()

    # 6a. Maximize ATPM ────────────────────────────────────────────────────────
    note("6a. Maximize ATPM (ATP maintenance reaction)")
    m2 = load_fresh()
    m2.objective = {m2.reactions.get_by_id("ATPM"): 1.0}
    m2.objective.direction = "max"
    sol_atpm_max = m2.optimize()
    result("objective.atpm_maximize.status",    sol_atpm_max.status)
    result("objective.atpm_maximize.objective", round(sol_atpm_max.objective_value, 6))

    # 6b. Minimize ATPM ────────────────────────────────────────────────────────
    note("\n6b. Minimize ATPM")
    m3 = load_fresh()
    m3.objective = {m3.reactions.get_by_id("ATPM"): 1.0}
    m3.objective.direction = "min"
    sol_atpm_min = m3.optimize()
    result("objective.atpm_minimize.status",    sol_atpm_min.status)
    result("objective.atpm_minimize.objective", round(sol_atpm_min.objective_value, 6))
    note("Minimum ATPM = its lower bound (check model: ATPM.lower_bound)")
    result("objective.atpm_lb", m.reactions.get_by_id("ATPM").lower_bound)

    # 6c. Minimize biomass ─────────────────────────────────────────────────────
    note("\n6c. Minimize biomass objective")
    m4 = load_fresh()
    m4.objective.direction = "min"
    sol_bio_min = m4.optimize()
    result("objective.biomass_minimize.status",    sol_bio_min.status)
    result("objective.biomass_minimize.objective", round(sol_bio_min.objective_value, 6))
    passed(sol_bio_min.objective_value < TOLERANCE,
           "Minimized biomass ≈ 0 (lower bound is 0)")

    # 6d. Multi-reaction weighted objective ───────────────────────────────────
    note("\n6d. Multi-reaction objective: {ATPM: 1.0, EX_o2_e: 2.0} maximize")
    m5 = load_fresh()
    atpm_rxn  = m5.reactions.get_by_id("ATPM")
    o2_rxn    = m5.reactions.get_by_id("EX_o2_e")
    m5.objective = {atpm_rxn: 1.0, o2_rxn: 2.0}
    m5.objective.direction = "max"
    sol_multi = m5.optimize()
    result("objective.multi.status",    sol_multi.status)
    result("objective.multi.objective", round(sol_multi.objective_value, 6))
    result("objective.multi.ATPM_flux", round(sol_multi.fluxes["ATPM"], 6))
    result("objective.multi.EX_o2_e_flux",
           round(sol_multi.fluxes["EX_o2_e"], 6),
           "EX_o2_e is an uptake; maximize = maximize O2 consumption magnitude?")

# ══════════════════════════════════════════════════════════════════════════════
# 7. MODEL INSPECTION REFERENCE VALUES
# ══════════════════════════════════════════════════════════════════════════════

def run_inspection():
    section("7. MODEL INSPECTION REFERENCE VALUES")
    m = load_fresh()

    # 7a. Model-level metadata ─────────────────────────────────────────────────
    note("7a. Model metadata")
    result("inspection.model_id",        m.id)
    result("inspection.n_reactions",     len(m.reactions))
    result("inspection.n_metabolites",   len(m.metabolites))
    result("inspection.n_genes",         len(m.genes))
    result("inspection.compartments",    list(m.compartments.keys()))

    # 7b. First 10 reaction IDs ────────────────────────────────────────────────
    note("\n7b. First 10 reactions (sorted model order)")
    first10 = [r.id for r in list(m.reactions)[:10]]
    result("inspection.first_10_reactions", first10)

    # 7c. PFK reaction detail ──────────────────────────────────────────────────
    note("\n7c. PFK reaction detail")
    pfk = m.reactions.get_by_id("PFK")
    result("inspection.PFK.name",          pfk.name)
    result("inspection.PFK.lower_bound",   pfk.lower_bound)
    result("inspection.PFK.upper_bound",   pfk.upper_bound)
    result("inspection.PFK.gpr",           pfk.gene_reaction_rule)
    result("inspection.PFK.reaction_str",  pfk.reaction)
    result("inspection.PFK.stoichiometry",
           {m.id: round(v, 4) for m, v in pfk.metabolites.items()})

    # 7d. Metabolite: glc__D_e ─────────────────────────────────────────────────
    note("\n7d. Metabolite glc__D_e detail")
    glc = m.metabolites.get_by_id("glc__D_e")
    result("inspection.glc__D_e.name",        glc.name)
    result("inspection.glc__D_e.formula",     glc.formula)
    result("inspection.glc__D_e.compartment", glc.compartment)
    result("inspection.glc__D_e.charge",      glc.charge)
    result("inspection.glc__D_e.reactions",
           [r.id for r in glc.reactions])

    # 7e. Gene b4025 detail ────────────────────────────────────────────────────
    note("\n7e. Gene b4025 detail")
    g = m.genes.get_by_id("b4025")
    result("inspection.b4025.name",      g.name)
    result("inspection.b4025.reactions", [r.id for r in g.reactions])

    # 7f. EX_glc__D_e bounds ──────────────────────────────────────────────────
    note("\n7f. EX_glc__D_e exchange reaction bounds")
    ex = m.reactions.get_by_id("EX_glc__D_e")
    result("inspection.EX_glc__D_e.lower_bound", ex.lower_bound)
    result("inspection.EX_glc__D_e.upper_bound", ex.upper_bound)
    note("Negative lower bound = glucose is taken up (by convention).")

# ══════════════════════════════════════════════════════════════════════════════
# 8. MODIFIED BOUNDS CROSS-CHECK
# ══════════════════════════════════════════════════════════════════════════════

def run_bounds():
    section("8. MODIFIED BOUNDS CROSS-CHECK")
    m = load_fresh()

    glucose_levels = [-10, -5, -2, -1, 0]
    note(f"FBA across glucose uptake rates: {glucose_levels}")
    bounds_results = {}
    for lb in glucose_levels:
        with m:
            m.reactions.get_by_id("EX_glc__D_e").lower_bound = lb
            sol = m.optimize()
            growth = round(sol.objective_value, 6) if sol.status == "optimal" else 0.0
            bounds_results[lb] = {"growth": growth, "status": sol.status}
            result(f"bounds.EX_glc__D_e_lb_{lb}.growth", growth)
    note("Growth should decrease monotonically as glucose uptake is restricted.")
    growths = [v["growth"] for v in bounds_results.values()]
    passed(growths == sorted(growths, reverse=True),
           "Growth decreases monotonically as glucose uptake is restricted (lb: -10 to 0)")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

SECTIONS = {
    "fba":        run_fba,
    "fva":        run_fva,
    "gene_ko":    run_gene_ko,
    "rxn_ko":     run_rxn_ko,
    "sampling":   run_sampling,
    "objectives": run_objectives,
    "inspection": run_inspection,
    "bounds":     run_bounds,
}

def main():
    global JSON_MODE
    parser = argparse.ArgumentParser(
        description="COBRApy ground-truth reference for Automata agent validation"
    )
    parser.add_argument(
        "--section",
        choices=list(SECTIONS.keys()),
        default=None,
        help="Run only this section (default: all)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print all results as JSON to stdout after running",
    )
    args = parser.parse_args()
    JSON_MODE = args.json

    _print("\n╔══════════════════════════════════════════════════════════╗")
    _print("║   Automata — COBRApy Ground Truth Reference Script       ║")
    _print(f"║   Model: {MODEL_ID:<49}║")
    _print(f"║   COBRApy version: {cobra.__version__:<39}║")
    _print("╚══════════════════════════════════════════════════════════╝")

    # Verify model loads before running any section
    try:
        _test = load_fresh()
        _print(f"\n  Model loaded: {_test.id} "
              f"({len(_test.reactions)} rxns, "
              f"{len(_test.metabolites)} mets, "
              f"{len(_test.genes)} genes)\n")
    except Exception as e:
        sys.exit(f"\n[ERROR] Could not load model '{MODEL_ID}': {e}")

    if args.section:
        SECTIONS[args.section]()
    else:
        for fn in SECTIONS.values():
            fn()

    _print("\n" + "═" * 60)
    _print("  All sections complete.")
    _print("  Use these values to validate agent responses.")
    _print("  Tolerance for float comparisons: ±1e-4")
    _print("═" * 60 + "\n")

    if args.json:
        print(json.dumps(RESULTS, indent=2, default=str))


if __name__ == "__main__":
    main()
