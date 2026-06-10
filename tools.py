from llama_index.core.tools import FunctionTool
from cobra.flux_analysis import flux_variability_analysis
from cobra.flux_analysis import single_gene_deletion, double_gene_deletion, single_reaction_deletion, double_reaction_deletion
from cobra.flux_analysis.parsimonious import pfba
from cobra.flux_analysis.geometric import geometric_fba
from cobra.sampling import OptGPSampler, ACHRSampler
from cobra.io import write_sbml_model
from difflib import SequenceMatcher
import multiprocessing
import psutil
import os

return_direct = True
model_manager = None
session_dir = None
session_uploads_dir = None

def set_model_manager(manager):
    global model_manager
    model_manager = manager

def set_session_dir(path):
    global session_dir
    session_dir = str(path)

def set_session_uploads_dir(path):
    global session_uploads_dir
    session_uploads_dir = str(path)

def get_current_model_id() -> str:
    """
    Returns the current model ID from the ModelManager.
    This function simulates fetching a model from a database or API.
    """
    if not model_manager or not model_manager.current_model_id:
        return {"error": "No model is currently loaded. Please load a model first."}
    
    return {"model_id": model_manager.current_model_id}
def check_model_loaded():
    """
    Checks if a model is currently loaded in the ModelManager.
    Raises an exception if no model is loaded.
    """
    if not model_manager or not model_manager.current_model_id:
        return {"error" : "No model is currently loaded. Please load a model first."}
    return {"response": "Model is loaded", "model_id": model_manager.current_model_id}
def load_model(model_id: str) -> str:
    """
    Loads a model by its ID from the ModelManager.
    This function simulates fetching a model from a database or API.
    """
    try:
        loaded_id = model_manager.load_model_by_id(model_id)
        if session_uploads_dir:
            save_path = os.path.join(session_uploads_dir, f"{loaded_id}.xml")
            write_sbml_model(model_manager.models[loaded_id], save_path)
        return {"response": f"Model '{loaded_id}' loaded successfully.", "model_id": loaded_id}
    except Exception as e:
        return {"error": str(e)}
def model_data() -> dict:
    """
    Returns metadata for a given a model.
    This function simulates fetching metadata from a database or API.
    """
    try:
        model = model_manager.get_current_model()

        for rxn in model.reactions:
            if rxn.lower_bound is None:
                rxn.lower_bound = -1000.0
            if rxn.upper_bound is None:
                rxn.upper_bound = 1000.0

        data = {
            "model_id": str(model.id),
            "objective_reaction": str(model.objective.expression) if model.objective.expression else "Not Set Yet",
            "reactions_count": len(model.reactions),
            "metabolites_count": len(model.metabolites),
            "genes_count": len(model.genes),
            "groups_count": len(model.groups),
            "compartments_count": len(model.compartments),
            "Compartments": str(list(model.compartments.values())),
        }
        return data

    except Exception as e:
        return {
            "error": str(e),
            "model_id": model_manager.current_model_id
        }
def model_info(query: str, count=10) -> dict:
    """
    Returns specific information for a given model based on a query.
    """
    try:
        model = model_manager.get_current_model()
    except:
        return {
            "error": f"No Model is found.",
        }
    try:
        if query == "reactions":
            return {"reactions": [f"{rxn.id} ({rxn.name})" for rxn in model.reactions][:count]}
        elif query == "genes":
            return {"genes": [f"{gn.id} ({gn.name})" for gn in model.genes][:count]}
        elif query == "metabolites":
            return {"metabolites": [f"{mb.id} ({mb.name})" for mb in model.metabolites][:count]}
        else:
            return {
                "error": f"Unknown query: {query}",
            }
    except Exception as e:
        return {
            "error": str(e),
            "model_id": model_manager.current_model_id
        }
def _fuzzy_score(query, text):
    return SequenceMatcher(None, query, text).ratio()

def _find_reaction(model, query: str):
    q = query.strip().lower()
    for r in model.reactions:
        if r.id.lower() == q:
            return ("exact", r)
    for r in model.reactions:
        if r.name.lower() == q:
            return ("exact", r)
    scored = sorted(
        model.reactions,
        key=lambda r: max(_fuzzy_score(q, r.id.lower()), _fuzzy_score(q, r.name.lower())),
        reverse=True
    )
    return ("fuzzy", scored[:5])

def _find_metabolite(model, query: str):
    q = query.strip().lower()
    for m in model.metabolites:
        if m.id.lower() == q:
            return ("exact", m)
    for m in model.metabolites:
        if m.name.lower() == q:
            return ("exact", m)
    scored = sorted(
        model.metabolites,
        key=lambda m: max(_fuzzy_score(q, m.id.lower()), _fuzzy_score(q, m.name.lower())),
        reverse=True
    )
    return ("fuzzy", scored[:5])

def _find_gene(model, query: str):
    q = query.strip().lower()
    for g in model.genes:
        if g.id.lower() == q:
            return ("exact", g)
    for g in model.genes:
        if g.name.lower() == q:
            return ("exact", g)
    scored = sorted(
        model.genes,
        key=lambda g: max(_fuzzy_score(q, g.id.lower()), _fuzzy_score(q, g.name.lower())),
        reverse=True
    )
    return ("fuzzy", scored[:5])

def reaction_info(rxn_id: str) -> dict:
    """
    Returns information about a specific reaction in the model.
    """
    try:
        model = model_manager.get_current_model()
        match_type, result = _find_reaction(model, rxn_id)
        if match_type == "exact":
            return {
                "Reaction id": result.id,
                "name": result.name,
                "Stochiometry": result.build_reaction_string(),
                "GPR": str(result.gpr) or "Not Set",
                "lower_bound": result.lower_bound,
                "upper_bound": result.upper_bound,
            }
        else:
            return {
                "message": f"No exact match for '{rxn_id}'. Did you mean?",
                "search_results": [{"id": r.id, "name": r.name} for r in result]
            }
    except Exception as e:
        return {"error": str(e)}
def metabolite_info(mb_id: str) -> dict:
    """
    Returns information about a specific metabolite in the model.
    """
    try:
        model = model_manager.get_current_model()
        match_type, result = _find_metabolite(model, mb_id)
        if match_type == "exact":
            return {
                "Metabolite id": result.id,
                "name": result.name,
                "Formula": result.formula,
                "Compartment": result.compartment,
                "Total Reactions": len(result.reactions),
                "Reactions": ', '.join([rxn.id for rxn in result.reactions]),
            }
        else:
            return {
                "message": f"No exact match for '{mb_id}'. Did you mean?",
                "search_results": [{"id": m.id, "name": m.name} for m in result]
            }
    except Exception as e:
        return {"error": str(e)}
def gene_info(gn_id: str) -> dict:
    """
    Returns information about a specific gene in the model.
    """
    try:
        model = model_manager.get_current_model()
        match_type, result = _find_gene(model, gn_id)
        if match_type == "exact":
            return {
                "Gene ID": result.id,
                "name": result.name,
                "Total Reactions": len(result.reactions),
                "Reactions": ', '.join([rxn.id for rxn in result.reactions]),
            }
        else:
            return {
                "message": f"No exact match for '{gn_id}'. Did you mean?",
                "search_results": [{"id": g.id, "name": g.name} for g in result]
            }
    except Exception as e:
        return {"error": str(e)}




def run_fba() -> str:
    """
    Performs Flux Balance Analysis (FBA) on the current metabolic model.
    Returns Objective value and Model Status.
    """
    try:
        model = model_manager.get_current_model()
    except Exception as e:
        return {"error": str(e)}

    if not model.objective.expression:
        return {"error": "No objective function is set on the model."}

    bounds = model_manager.bounds_data
    if bounds:
        try:
            for rxn_id, lval, uval in bounds:
                if model.reactions.has_id(rxn_id):
                    model.reactions.get_by_id(rxn_id).bounds = (lval, uval)
        except Exception:
            return {"error": "Wrong reaction bounds given in the uploaded CSV."}

    solution = model.optimize()
    model_manager.objective = solution.objective_value
    return {
        "Objective value": str(solution.objective_value),
        "status": str(solution.status),
    }

def run_pfba(fraction_of_optimum: float = 1.0) -> dict:
    """
    Performs Parsimonious FBA (pFBA) on the current metabolic model.
    Maximises the objective while minimising total absolute flux, reflecting
    the cell's preference for enzyme-efficient solutions.
    """
    try:
        model = model_manager.get_current_model()
    except Exception as e:
        return {"error": str(e)}

    if not model.objective.expression:
        return {"error": "No objective function is set on the model."}

    bounds = model_manager.bounds_data
    if bounds:
        try:
            for rxn_id, lval, uval in bounds:
                if model.reactions.has_id(rxn_id):
                    model.reactions.get_by_id(rxn_id).bounds = (lval, uval)
        except Exception:
            return {"error": "Wrong reaction bounds given in the uploaded CSV."}

    solution = pfba(model, fraction_of_optimum=fraction_of_optimum)

    if solution.status != "optimal":
        return {"error": f"pFBA did not reach an optimal solution (status: {solution.status})."}

    model_manager.objective = solution.objective_value
    total_flux = float(solution.fluxes.abs().sum())

    return {
        "Objective value": str(solution.objective_value),
        "Total absolute flux": str(round(total_flux, 6)),
        "fraction_of_optimum": fraction_of_optimum,
        "status": solution.status,
    }

def run_geometric_fba() -> dict:
    """
    Performs Geometric FBA (gFBA) on the current metabolic model.
    Returns a unique, centred flux distribution by iteratively bounding the
    convex hull of the feasible flux polytope.
    """
    try:
        model = model_manager.get_current_model()
    except Exception as e:
        return {"error": str(e)}

    if not model.objective.expression:
        return {"error": "No objective function is set on the model."}

    bounds = model_manager.bounds_data
    if bounds:
        try:
            for rxn_id, lval, uval in bounds:
                if model.reactions.has_id(rxn_id):
                    model.reactions.get_by_id(rxn_id).bounds = (lval, uval)
        except Exception:
            return {"error": "Wrong reaction bounds given in the uploaded CSV."}

    try:
        solution = geometric_fba(model)
    except RuntimeError as e:
        return {"error": f"Geometric FBA did not converge: {e}"}

    if solution.status != "optimal":
        return {"error": f"Geometric FBA did not reach an optimal solution (status: {solution.status})."}

    model_manager.objective = solution.objective_value
    return {
        "Objective value": str(solution.objective_value),
        "status": solution.status,
    }

def set_model_objective(reaction_id: str, direction: str = "max"):
    """
    Sets the objective on the current model to the given reaction ID.
    If the exact reaction ID is not found, falls back to fuzzy matching and
    uses the closest match.
    """
    try:
        model = model_manager.get_current_model()
        match_type, result = _find_reaction(model, reaction_id)
        if match_type == "fuzzy":
            rxn_obj = result[0]
        else:
            rxn_obj = result
        model.objective = {rxn_obj: 1.0}
        model.objective.direction = direction.lower()
        return {
            "status": "Objective set successfully.",
            "matched_reaction": rxn_obj.id,
            "match_type": match_type,
            "objective": str(model.objective.expression),
            "direction": model.objective.direction,
        }
    except Exception as e:
        return {"error": str(e)}  
def run_fva(rxn_names=None, fraction_of_optimum=0.9):
    """
    Runs Flux Variability Analysis (FVA) on the model given a Reaction List and a Fraction of Optimum (FO) Value.
    """
    try:
        model = model_manager.get_current_model()
        rxn_obj_list = []

        if not model.objective.expression:
            return {"error": "No Objective Function is set for the model."}

        if not rxn_names:
            rxn_obj_list = list(model.reactions)
        else:
            for name in rxn_names:
                match = next(
                    (rxn for rxn in model.reactions if rxn.id.lower() == name.lower()),
                    None
                )
                if match is None:
                    match = next(
                        (rxn for rxn in model.reactions if name.lower() in rxn.name.lower()),
                        None
                    )
                if match is None:
                    raise ValueError(f"Reaction '{name}' not found in model.")
                rxn_obj_list.append(match)

        fva_result = flux_variability_analysis(model, rxn_obj_list, fraction_of_optimum=fraction_of_optimum)

        fva_df = fva_result.reset_index()
        fva_df.insert(0, "Reaction Name", [rxn.name for rxn in rxn_obj_list])
        fva_df.columns = ["Reaction Name", "Reaction ID", "Maximum Flux", "Minimum Flux"]

        if len(fva_df) > 5:
            output_dir = os.path.join(session_dir, 'fva')
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, "fva_result.csv")
            fva_df.to_csv(csv_path, index=False)

            return {
                "fraction_of_optimum": fraction_of_optimum,
                "fva_output": f"First 5 rows: {fva_df.iloc[:5,:].to_dict(orient='records')}",
                "message": f"FVA result has {len(fva_df)} entries, saved to {csv_path} as CSV file."
            }
        else:
            return {
                "fraction_of_optimum": fraction_of_optimum,
                "fva_output": fva_df.to_dict(orient="records")
            }

    except Exception as e:
        return {"error": str(e)}
def gene_knockout_simulation(gene_names, type: str) -> dict:
    """
    Performs single or double gene knockout simulations on the loaded metabolic model.
    """
    try:
        if isinstance(gene_names, str):
            gene_names = [g.strip() for g in gene_names.split(",") if g.strip()]
        model = model_manager.get_current_model()
        valid_genes = []
        seen = set()
        for name in gene_names:
            tag, payload = _find_gene(model, name)
            gene = payload if tag == "exact" else (payload[0] if payload else None)
            if gene and gene.id not in seen:
                valid_genes.append(gene)
                seen.add(gene.id)

        if not valid_genes:
            return {"error": "None of the provided genes are valid in this model."}

        if type == "single":
            result = single_gene_deletion(model, gene_list=valid_genes)
        elif type == "double":
            result = double_gene_deletion(model, gene_list=valid_genes)
        else:
            return {"error": "Invalid type. Choose 'single' or 'double'."}

        result = result.rename(columns={
            "growth": "Post-KO Growth",
            "ids": "Gene(s)",
            "status": "Solver Status"
        })

        if len(result) > 5:
            file_path = os.path.join(session_dir, "knockouts", "gene_knockout_result.csv")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            result.to_csv(file_path, index=False)
            subset = result.iloc[:5, :5]
            return {"file": file_path, "data": subset.to_dict(orient="records"), "note": "Too many results to display. Download CSV."}

        return result.to_dict(orient="records")

    except Exception as e:
        return {"error": str(e)}
def reaction_knockout_simulation(reaction_names, type: str) -> dict:
    """
    Performs single or double reaction knockout simulations on the loaded metabolic model.
    """
    try:
        if isinstance(reaction_names, str):
            reaction_names = [r.strip() for r in reaction_names.split(",") if r.strip()]
        model = model_manager.get_current_model()
        valid_rxns = []
        seen = set()
        for name in reaction_names:
            tag, payload = _find_reaction(model, name)
            rxn = payload if tag == "exact" else (payload[0] if payload else None)
            if rxn and rxn.id not in seen:
                valid_rxns.append(rxn)
                seen.add(rxn.id)

        if not valid_rxns:
            return {"error": "None of the provided reactions are valid in this model."}

        if type == "single":
            result = single_reaction_deletion(model, reaction_list=valid_rxns)
        elif type == "double":
            result = double_reaction_deletion(model, reaction_list=valid_rxns)
        else:
            return {"error": "Invalid type. Choose 'single' or 'double'."}

        result = result.rename(columns={
            "growth": "Post-KO Growth",
            "ids": "Reaction(s)",
            "status": "Solver Status"
        })

        if len(result) > 5:
            file_path = os.path.join(session_dir, "knockouts", "reaction_knockout_result.csv")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            result.to_csv(file_path, index=False)
            subset = result.iloc[:5, :5]
            return {"file": file_path, "data": subset.to_dict(orient="records"), "note": "Too many results to display. Download CSV."}

        return result.to_dict(orient="records")

    except Exception as e:
        return {"error": str(e)}
        
def recommend_sampling_config(model):
    n_rxns = len(model.reactions)
    cpu_cores = multiprocessing.cpu_count()
    ram_gb = psutil.virtual_memory().total / 1e9

    if n_rxns < 500:
        method = "achr"
    elif cpu_cores >= 4:
        method = "optgp"
    else:
        method = "achr"
        
    thinning = max(10, int(n_rxns / 5))
    thinning = min(thinning, 500)

    processes = None
    if method == "optgp":
        if cpu_cores >= 8:
            processes = cpu_cores // 2
        elif cpu_cores >= 4:
            processes = 2
        else:
            processes = 1
        if ram_gb < 4:
            processes = min(processes, 2)

    return {
        "method": method,
        "thinning": thinning,
        "processes": processes
    }
def sample_metabolic_model(reaction_count=1000):
    """
    Samples a metabolic model given the number of samples.
    """
    model = model_manager.get_current_model()
    # error handling
    config = recommend_sampling_config(model)
    config["method"] = model_manager.sampler
    
    if config["method"] == "achr":
        sampler = ACHRSampler(model, thinning=config["thinning"])
    else:
        sampler = OptGPSampler(model, thinning=config["thinning"], processes=config["processes"])
    samples = sampler.sample(reaction_count)
    subset = samples.iloc[:5, :5]
    sampling_dir = os.path.join(session_dir, 'flux_sampling')
    os.makedirs(sampling_dir, exist_ok=True)
    csv_path = os.path.join(sampling_dir, 'flux_sampling_result.csv')
    samples.to_csv(csv_path, index=False)
    return {
        "status": "success",
        "n_samples": reaction_count,
        "method": config["method"],
        "thinning": config["thinning"],
        "processes": config["processes"],
        "save_path": csv_path,
        "samples": subset.to_dict(orient="records")
    }


def list_escher_maps() -> dict:
    """
    Returns all available Escher metabolic map names.
    """
    try:
        import escher
        maps = escher.list_available_maps()
        names = [m["map_name"] for m in maps]
        return {"response": names}
    except Exception as e:
        return {"error": str(e)}

def visualize_escher(map_name: str = None) -> dict:
    """
    Runs FBA on the current model and renders a flux map using Escher, saved as an HTML file.
    Automatically selects the best matching Escher map for the loaded model.
    Optionally accepts a map_name to override auto-detection.
    """
    try:
        import escher

        model = model_manager.get_current_model()
        if model is None:
            return {"error": "No model loaded. Please load a model first."}

        if not model.objective.expression:
            return {"error": "No objective function is set on the model. Please set an objective before visualizing."}

        # Auto-detect map from model ID if not specified
        if map_name is None:
            available = escher.list_available_maps()
            model_id = model_manager.current_model_id
            matches = [m["map_name"] for m in available if m["map_name"].startswith(model_id + ".")]
            if not matches:
                all_names = [m["map_name"] for m in available]
                return {
                    "error": f"No Escher map found for model '{model_id}'. Use list_escher_maps to see all available maps and pass one as map_name.",
                    "available_maps": all_names
                }
            # Prefer "Central metabolism" if present, otherwise take first match
            map_name = next((m for m in matches if "Central metabolism" in m), matches[0])

        # Apply bounds if loaded, then run FBA
        bounds = model_manager.bounds_data
        if bounds:
            try:
                for rxn_id, lval, uval in bounds:
                    if model.reactions.has_id(rxn_id):
                        model.reactions.get_by_id(rxn_id).bounds = (lval, uval)
            except Exception:
                return {"error": "Wrong reaction bounds given in the uploaded CSV."}

        solution = model.optimize()
        if solution.status != "optimal":
            return {"error": f"FBA did not reach an optimal solution (status: {solution.status})."}

        reaction_data = solution.fluxes.to_dict()

        # Save HTML to session output directory
        escher_dir = os.path.join(session_dir, "escher")
        os.makedirs(escher_dir, exist_ok=True)
        safe_name = map_name.replace("/", "_").replace(" ", "_")
        filepath = os.path.join(escher_dir, f"{safe_name}_flux.html")

        builder = escher.Builder(map_name=map_name, model=model, reaction_data=reaction_data)
        builder.save_html(filepath)

        return {
            "response": f"Escher flux map saved to {filepath}. Map: '{map_name}'. Objective value: {solution.objective_value:.4f}",
            "file": filepath,
            "map_name": map_name,
            "objective_value": solution.objective_value
        }
    except Exception as e:
        return {"error": str(e)}


def run_memote_report() -> dict:
    """
    Run the memote quality test suite on the currently loaded metabolic model.
    Returns a concise pass/fail summary; also saves a full HTML report to the
    session directory.
    """
    try:
        model = model_manager.get_current_model()
    except Exception as e:
        return {"error": f"No model loaded: {e}"}

    try:
        from memote.suite.api import test_model, snapshot_report
    except ImportError:
        return {"error": "memote is not installed. Run: pip install memote"}

    import contextlib, io as _io

    try:
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exit_code, result = test_model(
                model,
                results=True,
                pytest_args=["--tb", "no", "-q", "--no-header"],
            )

        cases = result.cases  # {test_name: {title, result, message, data, metric, ...}}

        passed, failed, skipped = [], [], []

        for test_name, case in cases.items():
            outcome = case.get("result", "unknown")
            title   = case.get("title", test_name)
            message = case.get("message") or ""

            # Parametrized tests store result as {param: outcome}
            if isinstance(outcome, dict):
                outcomes = list(outcome.values())
                n_fail = sum(1 for o in outcomes if o in ("failed", "error"))
                if n_fail:
                    outcome = "failed"
                    message = message or f"{n_fail}/{len(outcomes)} variants failed"
                elif all(o == "passed" for o in outcomes):
                    outcome = "passed"
                else:
                    outcome = "skipped"

            if outcome == "passed":
                passed.append(title)
            elif outcome in ("failed", "error"):
                failed.append({"title": title, "message": message[:250]})
            else:
                skipped.append(title)

        # Save full HTML report
        report_path = None
        try:
            html = snapshot_report(result)
            report_dir = os.path.join(session_dir, "memote")
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, "memote_report.html")
            with open(report_path, "w", encoding="utf-8") as fh:
                fh.write(html)
        except Exception:
            pass  # report generation failure is non-fatal

        total_run = len(passed) + len(failed)

        def _fmt(f):
            return f"{f['title']}: {f['message']}" if f["message"] else f["title"]

        base = {
            "tests_passed": len(passed),
            "tests_failed": len(failed),
            "tests_skipped": len(skipped),
        }
        if report_path:
            base["full_report"] = report_path

        if not failed:
            return {
                **base,
                "status": "passed",
                "summary": (
                    f"All {len(passed)} memote tests passed. "
                    "The model meets quality standards."
                ),
            }

        TOO_MANY = 10
        if len(failed) > TOO_MANY:
            return {
                **base,
                "status": "failed",
                "summary": (
                    f"{len(failed)} out of {total_run} tests failed "
                    f"(too many to list — showing the top {min(5, len(failed))} below)."
                ),
                "major_failures": [_fmt(f) for f in failed[:5]],
            }
        else:
            return {
                **base,
                "status": "failed",
                "summary": f"{len(failed)} out of {total_run} tests failed.",
                "failures": [_fmt(f) for f in failed],
            }

    except Exception as e:
        return {"error": str(e)}


def add_reaction(
    reaction_id: str,
    equation: str,
    reaction_name: str = "",
    lower_bound: float = 0.0,
    upper_bound: float = 1000.0,
    gene_reaction_rule: str = "",
    subsystem: str = "",
    new_metabolites_info: str = "",
) -> dict:
    """
    Creates a new reaction and adds it to the currently loaded model.

    equation             — e.g. '1.0 malACP_c + h_c --> co2_c + 1.0 ACP_c'
                           Use '-->' for irreversible, '<-->' for reversible.
                           Coefficients are optional (default 1.0).

    new_metabolites_info — comma-separated 'id|formula|name|compartment' entries
                           for metabolites NOT already in the model.
                           E.g. 'malACP_c|C25H39N2O10PRS|Malonyl-ACP|c'
    """
    import re as _re, cobra as _cobra

    try:
        model = model_manager.get_current_model()
    except Exception as e:
        return {"error": str(e)}

    if not _re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', reaction_id):
        return {"error": f"reaction_id '{reaction_id}' is not a valid SBML SId. Use letters/digits/underscores; must start with letter or _."}
    if model.reactions.has_id(reaction_id):
        return {"error": f"Reaction '{reaction_id}' already exists in the model."}

    # _coerce in executor.py splits comma-containing strings into lists;
    # convert back to a single string so our parser can handle it normally.
    if isinstance(new_metabolites_info, list):
        new_metabolites_info = ", ".join(new_metabolites_info)

    # Parse new_metabolites_info → {met_id: {formula, name, compartment}}
    new_met_details: dict = {}
    if new_metabolites_info.strip():
        for entry in new_metabolites_info.split(","):
            fields = [f.strip() for f in entry.strip().split("|")]
            if not fields[0]:
                continue
            new_met_details[fields[0]] = {
                "formula":     fields[1] if len(fields) > 1 else None,
                "name":        fields[2] if len(fields) > 2 else "",
                "compartment": fields[3] if len(fields) > 3 else None,
            }

    if not isinstance(equation, str) or not equation.strip():
        return {"error": "equation is required."}

    # Detect reversibility and split into lhs/rhs.
    # Check compound separators before simple '=' to avoid splitting inside '<=>'
    if "<-->" in equation or "<=>" in equation:
        sep = "<-->" if "<-->" in equation else "<=>"
        reversible = True
    elif "-->" in equation:
        sep = "-->"
        reversible = False
    elif "=>" in equation:
        sep = "=>"
        reversible = False
    elif "=" in equation:
        sep = "="
        reversible = False
    else:
        return {"error": "equation must contain '-->' or '=>' (irreversible) or '<-->' or '<=>' (reversible) as separator."}

    sides = equation.split(sep, 1)
    if len(sides) != 2:
        return {"error": f"Could not split equation on '{sep}'."}
    lhs, rhs = sides

    def _parse_side(text: str, sign: float) -> dict:
        result = {}
        for term in text.split("+"):
            term = term.strip()
            if not term:
                continue
            # Require whitespace (or parens) between coefficient and metabolite ID so that
            # "2of" is treated as met_id "2of" with coeff 1, not coeff 2 + met_id "of".
            m_space = _re.match(r'^(\d+\.?\d*|\d*\.\d+)\s+(.+)$', term)
            m_paren = _re.match(r'^(\d+\.?\d*|\d*\.\d+)\((.+)\)$', term)
            if m_space:
                coeff = float(m_space.group(1))
                mid = m_space.group(2).strip()
            elif m_paren:
                coeff = float(m_paren.group(1))
                mid = m_paren.group(2).strip()
            else:
                coeff = 1.0
                mid = term.strip("() ")
            if mid:
                result[mid] = sign * coeff
        return result

    stoich_map: dict = {}
    stoich_map.update(_parse_side(lhs, -1.0))
    stoich_map.update(_parse_side(rhs, +1.0))

    if not stoich_map:
        return {"error": "No metabolites found in equation. Check the format."}

    def _infer_compartment(mid: str) -> str:
        if "_" in mid:
            suffix = mid.rsplit("_", 1)[-1]
            if 1 <= len(suffix) <= 2:
                return suffix
        return "c"

    metabolite_dict: dict = {}
    reused_ids, created_ids = [], []

    for mid, coeff in stoich_map.items():
        if model.metabolites.has_id(mid):
            metabolite_dict[model.metabolites.get_by_id(mid)] = coeff
            reused_ids.append(mid)
        else:
            details = new_met_details.get(mid, {})
            comp = details.get("compartment") or _infer_compartment(mid)
            met_obj = _cobra.Metabolite(
                id=mid,
                formula=details.get("formula"),
                name=details.get("name", ""),
                compartment=comp,
            )
            metabolite_dict[met_obj] = coeff
            created_ids.append(mid)

    # Apply reversibility default for bounds
    eff_lb = float(lower_bound)
    eff_ub = float(upper_bound)
    if reversible and eff_lb == 0.0:
        eff_lb = -1000.0

    rxn = _cobra.Reaction(
        id=reaction_id,
        name=reaction_name or reaction_id,
        subsystem=subsystem,
        lower_bound=eff_lb,
        upper_bound=eff_ub,
    )
    if gene_reaction_rule:
        rxn.gene_reaction_rule = gene_reaction_rule
    rxn.add_metabolites(metabolite_dict)
    model.add_reactions([rxn])

    result = {
        "status": "success",
        "reaction_id": reaction_id,
        "reaction_name": rxn.name,
        "equation": rxn.build_reaction_string(),
        "lower_bound": rxn.lower_bound,
        "upper_bound": rxn.upper_bound,
        "gene_reaction_rule": rxn.gene_reaction_rule or "",
        "metabolites_reused_from_model": reused_ids,
    }
    if created_ids:
        result["metabolites_created_new"] = created_ids
        result["note"] = (
            "New metabolites were created. "
            "Those without formula/name/compartment details were stored minimally with inferred compartment."
        )
    return result


def set_reaction_bounds(
    reaction_id: str = None,
    lower_bound: float = None,
    upper_bound: float = None,
    csv_path: str = None,
) -> dict:
    """
    Set or change flux bounds on one or more reactions in the loaded model.

    Single-reaction mode  — provide reaction_id, lower_bound, upper_bound.
    CSV batch mode        — provide csv_path to a file with columns rxn_id, lb, ub.
    """
    import pandas as _pd, os as _os

    try:
        model = model_manager.get_current_model()
    except Exception as e:
        return {"error": str(e)}

    # ── CSV batch mode ──────────────────────────────────────────────────────────
    if csv_path is not None:
        if not _os.path.isabs(csv_path) and session_uploads_dir:
            csv_path = _os.path.join(session_uploads_dir, csv_path)
        if not _os.path.exists(csv_path):
            return {"error": f"CSV file not found: {csv_path}"}
        try:
            df = _pd.read_csv(csv_path)
        except Exception as e:
            return {"error": f"Could not read CSV: {e}"}
        required_cols = {"rxn_id", "lb", "ub"}
        if not required_cols.issubset(set(df.columns)):
            return {"error": f"CSV must have columns: rxn_id, lb, ub. Found: {list(df.columns)}"}

        updated, errors = [], []
        for _, row in df.iterrows():
            rid = str(row["rxn_id"]).strip()
            try:
                lb, ub = float(row["lb"]), float(row["ub"])
            except (ValueError, TypeError):
                errors.append({"rxn_id": rid, "reason": "non-numeric lb or ub"})
                continue
            match_type, result = _find_reaction(model, rid)
            rxn = result if match_type == "exact" else result[0]
            prev = rxn.bounds
            rxn.bounds = (lb, ub)
            updated.append({
                "rxn_id": rxn.id,
                "match_type": match_type,
                "queried_as": rid,
                "previous_bounds": list(prev),
                "new_bounds": [lb, ub],
            })

        return {
            "status": "success",
            "updated_count": len(updated),
            "updated": updated,
            "errors": errors,
        }

    # ── Single reaction mode ────────────────────────────────────────────────────
    if reaction_id is None:
        return {"error": "Provide either reaction_id (single mode) or csv_path (batch mode)."}
    if lower_bound is None or upper_bound is None:
        return {"error": "lower_bound and upper_bound are both required in single-reaction mode."}

    match_type, result = _find_reaction(model, reaction_id)

    if match_type == "exact":
        rxn = result
        note = None
    else:
        rxn = result[0]
        note = (
            f"No exact match for '{reaction_id}'. "
            f"Used closest match: '{rxn.id}' ({rxn.name}). "
            f"Other candidates: {[r.id for r in result[1:]]}"
        )

    prev = rxn.bounds
    rxn.bounds = (float(lower_bound), float(upper_bound))

    response = {
        "status": "success",
        "matched_reaction": rxn.id,
        "reaction_name": rxn.name,
        "match_type": match_type,
        "previous_bounds": list(prev),
        "new_bounds": [float(lower_bound), float(upper_bound)],
    }
    if note:
        response["fuzzy_note"] = note
    return response


# ── Planner metadata ──────────────────────────────────────────────────────────
# Each tool function used by the planner carries a _planner_meta attribute.
# planner/tool_registry.py reads these to build TOOL_FN_MAP, TOOL_USER_INPUTS,
# and AVAILABLE_TOOLS_TEXT — the single source of truth for all three.

load_model._planner_meta = {
    "name": "load_model",
    "description": "Load a metabolic model by ID or from a local SBML file",
    "params": {
        "model_id": {
            "description": "Model identifier (e.g. 'iHsa', 'e_coli_core') or path to a local SBML file (.xml/.sbml)",
            "default": None,
            "required": True,
        }
    },
}

run_fba._planner_meta = {
    "name": "run_fba",
    "description": "Run Flux Balance Analysis on the currently loaded model",
    "params": {},
}

run_fva._planner_meta = {
    "name": "run_fva",
    "description": "Run Flux Variability Analysis to compute min/max flux ranges",
    "params": {
        "rxn_names": {
            "description": "Specific reaction names to analyze. Leave empty to run FVA on all reactions.",
            "default": None,
            "required": False,
        },
        "fraction_of_optimum": {
            "description": "Fraction of the optimal objective value to maintain (default: 1.0)",
            "default": 1.0,
            "required": False,
        },
    },
}

gene_knockout_simulation._planner_meta = {
    "name": "gene_knockout_simulation",
    "description": "Simulate single or double gene knockouts and assess growth impact",
    "params": {
        "gene_names": {
            "description": "List of gene names to knock out (e.g. ['b0001', 'b0002'])",
            "default": None,
            "required": True,
        },
        "type": {
            "description": "Knockout type: 'single' for individual knockouts, 'double' for pairwise knockouts",
            "default": "single",
            "required": True,
        },
    },
}

reaction_knockout_simulation._planner_meta = {
    "name": "reaction_knockout_simulation",
    "description": "Simulate single or double reaction knockouts",
    "params": {
        "reaction_names": {
            "description": "List of reaction names to knock out",
            "default": None,
            "required": True,
        },
        "type": {
            "description": "Knockout type: 'single' or 'double'",
            "default": "single",
            "required": True,
        },
    },
}

sample_metabolic_model._planner_meta = {
    "name": "sample_metabolic_model",
    "description": "Sample flux distributions using ACHR or OptGP",
    "params": {
        "reaction_count": {
            "description": "Number of reactions to include in sampling. Leave empty to sample all reactions.",
            "default": None,
            "required": False,
        }
    },
}

set_model_objective._planner_meta = {
    "name": "set_model_objective",
    "description": "Set the objective function before running FBA",
    "params": {
        "reaction_id": {
            "description": "Reaction ID to use as objective (e.g. 'BIOMASS_Ec_core'). Fuzzy matched if not found exactly.",
            "default": None,
            "required": True,
        },
        "direction": {
            "description": "Optimization direction: 'max' or 'min'",
            "default": "max",
            "required": True,
        },
    },
}

visualize_escher._planner_meta = {
    "name": "visualize_escher",
    "description": "Run FBA and render the flux distribution on an Escher metabolic map (HTML). Auto-detects the map for the loaded model. Must follow any FBA or optimization step.",
    "params": {
        "map_name": {
            "description": "Escher map name to use (e.g. 'e_coli_core.Central metabolism'). Leave empty to auto-detect from the loaded model.",
            "default": None,
            "required": False,
        }
    },
}

list_escher_maps._planner_meta = {
    "name": "list_escher_maps",
    "description": "List all available Escher map names (use when auto-detection fails)",
    "params": {},
}

run_pfba._planner_meta = {
    "name": "run_pfba",
    "description": "Run Parsimonious FBA (pFBA) — maximises the objective while minimising total flux to find the most enzyme-efficient or resource-efficient solution",
    "params": {
        "fraction_of_optimum": {
            "description": "Fraction of the optimal objective value that must be maintained (default: 1.0). Lower values allow slightly sub-optimal but sparser solutions.",
            "default": 1.0,
            "required": False,
        }
    },
}

run_geometric_fba._planner_meta = {
    "name": "run_geometric_fba",
    "description": "Run Geometric FBA (gFBA) — returns a unique, centred flux distribution by iteratively bounding the convex hull of the feasible flux space",
    "params": {},
}

run_memote_report._planner_meta = {
    "name": "run_memote_report",
    "description": "Run the memote quality test suite on the loaded model and return a concise pass/fail summary with a full HTML report",
    "params": {},
}

add_reaction._planner_meta = {
    "name": "add_reaction",
    "description": "Add a new reaction (with stoichiometry, bounds, GPR, subsystem) to the currently loaded model",
    "params": {
        "reaction_id": {
            "description": "SBML-valid reaction ID (letters/digits/underscores, starts with letter or _). E.g. 'R_3OAS140'",
            "default": None,
            "required": True,
        },
        "equation": {
            "description": "Reaction equation. Format: 'coeff met_id + coeff met_id --> coeff met_id'. Use '-->' for irreversible, '<-->' for reversible. Coefficients optional (default 1). E.g. '1.0 malACP_c + h_c --> co2_c + 1.0 ACP_c'",
            "default": None,
            "required": True,
        },
        "reaction_name": {
            "description": "Human-readable reaction name (optional). E.g. '3-oxoacyl-ACP synthase (C14:0)'",
            "default": "",
            "required": False,
        },
        "lower_bound": {
            "description": "Lower flux bound (default 0.0 for irreversible; auto-set to -1000.0 if '<-->' used and not overridden)",
            "default": 0.0,
            "required": False,
        },
        "upper_bound": {
            "description": "Upper flux bound (default 1000.0)",
            "default": 1000.0,
            "required": False,
        },
        "gene_reaction_rule": {
            "description": "Boolean GPR expression (optional). E.g. '( STM2378 or STM1197 )'",
            "default": "",
            "required": False,
        },
        "subsystem": {
            "description": "Metabolic subsystem (optional). E.g. 'Fatty Acid Synthesis'",
            "default": "",
            "required": False,
        },
    },
}

set_reaction_bounds._planner_meta = {
    "name": "set_reaction_bounds",
    "description": "Set or change flux bounds on one or more reactions in the loaded model (single reaction or CSV batch)",
    "params": {
        "reaction_id": {
            "description": "Reaction ID or name to update (single-reaction mode). Fuzzy matched if no exact match.",
            "default": None,
            "required": False,
        },
        "lower_bound": {
            "description": "New lower flux bound for the reaction (single-reaction mode).",
            "default": None,
            "required": False,
        },
        "upper_bound": {
            "description": "New upper flux bound for the reaction (single-reaction mode).",
            "default": None,
            "required": False,
        },
        "csv_path": {
            "description": "Path to a CSV file (columns: rxn_id, lb, ub) to update bounds in batch. Relative paths resolve against the session uploads directory.",
            "default": None,
            "required": False,
        },
    },
}


######### TOOL SETUP

check_load_model_tool = FunctionTool.from_defaults(
    fn=check_model_loaded,
    name="check_model_loaded",
    description="Checks if a model is currently loaded in the ModelManager. Returns the model ID if loaded, otherwise an error message.",
    return_direct=return_direct
)
current_model_tool = FunctionTool.from_defaults(
    fn=get_current_model_id,
    name="get_current_model_id",
    description="""Returns the current model ID from the ModelManager.""",
    return_direct=return_direct
)
load_model_tool = FunctionTool.from_defaults(
    fn=load_model,
    name="load_model",
    description=(
        "Loads a metabolic model by its BiGG or BioModels ID. "
        "Use the plain model ID exactly as it appears in the database — for example: "
        "'e_coli_core', 'iJN1463', 'iML1515'. "
        "Do NOT use prefixes like 'BIGG:' or 'BioModels:' — just pass the bare ID string."
    ),
    return_direct=return_direct
)
model_data_tool = FunctionTool.from_defaults(
    fn=model_data,
    name="model_data",
    description="Returns the metadata related to a given model: model_id, model_name, reactions_count, metabolites_count, genes_count",
    return_direct=return_direct
)
model_info_tool = FunctionTool.from_defaults(
    fn=model_info,
    name="model_info",
    description="""Returns categorical data for a given model based on a query.
    Queries can be 'reactions', 'genes', or 'metabolites'.
    You can also specify the number of items to return with the 'count' parameter.""",
    return_direct=return_direct
)
reaction_info_tool = FunctionTool.from_defaults(
    fn=reaction_info,
    name="reaction_info",
    description="Returns the metadata related to a given Reaction by ID: reaction_id, name, Stochiometry, GPR, lower_bound, upper_bound",
    return_direct=return_direct
)
metabolite_info_tool = FunctionTool.from_defaults(
    fn=metabolite_info,
    name="metabolite_info",
    description="Returns the metadata related to a given Metabolite by Name: metabolite_id, name, Formula, Compartment, Total Reactions, Reactions",
    return_direct=return_direct
)
gene_info_tool = FunctionTool.from_defaults(
    fn=gene_info,
    name="gene_info",
    description="Returns the metadata related to a given Gene by Name: gene_id, name, Total Reactions, Reactions",
    return_direct=return_direct
)
# Change the description
run_fba_tool = FunctionTool.from_defaults(
    fn=run_fba,
    name="run_flux_balance_analysis",
    description="Uses a bounds dictionary loaded using `set_reaction_bounds_for_FBA()` for Flux Balance Analysis (FBA).",
    return_direct=return_direct
)
set_objective_tool = FunctionTool.from_defaults(
    fn=set_model_objective,
    name="set_model_objective_value",
    description="Sets the objective for the current metabolic model given a dictonary of Reaction_id: coefficient pairs and an optional direction value.",
    return_direct=return_direct
)
run_fva_tool = FunctionTool.from_defaults(
    fn=run_fva,
    name="run_flux_variability_analysis",
    description="Runs Flux Variability Analysis (FVA) on the model. Optionally accepts a list of reaction names (rxn_names); if not provided or empty, runs FVA on all reactions in the model. Also accepts a fraction_of_optimum value (default 0.9).",
    return_direct=return_direct
)
gene_knockout_tool = FunctionTool.from_defaults(
    fn=gene_knockout_simulation,
    name="gene_knockout_simulation",
    description="Performs single or double gene knockout simulations on the loaded metabolic model",
    return_direct=return_direct
)
reaction_knockout_tool = FunctionTool.from_defaults(
    fn=reaction_knockout_simulation,
    name="reaction_knockout_simulation",
    description="Performs single or double reaction knockout simulations on the loaded metabolic model",
    return_direct=return_direct
)
flux_sampler_tool = FunctionTool.from_defaults(
    fn=sample_metabolic_model,
    name="sample_metabolic_model",
    description="Flux Sampling / Flux Sample Analysis a metabolic model given the number of samples.",
    return_direct=return_direct
)
list_escher_maps_tool = FunctionTool.from_defaults(
    fn=list_escher_maps,
    name="list_escher_maps",
    description="Lists all available Escher metabolic map names. Use this when auto-detection fails and the user needs to pick a specific map.",
    return_direct=return_direct
)
visualize_escher_tool = FunctionTool.from_defaults(
    fn=visualize_escher,
    name="visualize_escher",
    description="Runs FBA on the current model and renders a flux map using Escher, saved as an HTML file. Automatically selects the best map for the loaded model. Optionally accepts a map_name to override auto-detection (use list_escher_maps to find valid names).",
    return_direct=return_direct
)

memote_report_tool = FunctionTool.from_defaults(
    fn=run_memote_report,
    name="run_memote_quality_report",
    description=(
        "Runs the memote test suite on the currently loaded metabolic model. "
        "Returns a concise quality summary (passed/failed tests with messages) "
        "and saves a full HTML report to the session directory."
    ),
    return_direct=return_direct
)

run_pfba_tool = FunctionTool.from_defaults(
    fn=run_pfba,
    name="run_parsimonious_fba",
    description=(
        "Runs Parsimonious FBA (pFBA) on the currently loaded model. "
        "Maximises the objective while minimising total absolute flux, giving an "
        "enzyme-efficient solution. Optionally accepts fraction_of_optimum (default 1.0)."
    ),
    return_direct=return_direct
)

run_geometric_fba_tool = FunctionTool.from_defaults(
    fn=run_geometric_fba,
    name="run_geometric_fba",
    description=(
        "Runs Geometric FBA (gFBA) on the currently loaded model. "
        "Returns a unique, centred flux distribution by iteratively bounding "
        "the convex hull of the feasible flux polytope."
    ),
    return_direct=return_direct
)

add_reaction_tool = FunctionTool.from_defaults(
    fn=add_reaction,
    name="add_reaction_to_model",
    description=(
        "Adds a new reaction to the currently loaded metabolic model. "
        "Accepts a reaction ID, equation string (e.g. '1.0 A + B --> C'), "
        "optional reaction name, flux bounds, gene-reaction rule, and subsystem."
    ),
    return_direct=return_direct
)

set_reaction_bounds_tool = FunctionTool.from_defaults(
    fn=set_reaction_bounds,
    name="set_reaction_bounds",
    description=(
        "Set or change the flux bounds (lower_bound, upper_bound) for one or more "
        "reactions in the loaded model. "
        "Single-reaction mode: provide reaction_id, lower_bound, upper_bound — "
        "fuzzy matched if no exact ID/name match. "
        "CSV batch mode: provide csv_path to an uploaded CSV with columns rxn_id, lb, ub."
    ),
    return_direct=return_direct,
)






# TEST CODE

# model_manager = ModelManager()
# model_manager.load_model_by_id('e_coli_core')
# file_path = r"E:\INTERNSHIP\IITM\metabolic\uploads\bounds_data\e_coli_bounds.csv"
# model_manager.bounds_dict = pd.read_csv(file_path).values.tolist()
# print(run_fba())