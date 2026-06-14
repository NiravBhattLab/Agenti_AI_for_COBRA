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

ZERO_FLUX_THRESHOLD = 1e-6

def _clip_zero(value):
    """Suppress LP solver numerical noise: treat |x| < 1e-6 as exactly 0."""
    if value is None:
        return value
    try:
        return 0.0 if abs(float(value)) < ZERO_FLUX_THRESHOLD else float(value)
    except (TypeError, ValueError):
        return value

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
def model_info(query: str, count=20) -> dict:
    """
    Returns categorical data for a given model based on a query.
    """
    _MAX_COUNT = 30
    count = min(int(count), _MAX_COUNT)
    try:
        model = model_manager.get_current_model()
    except:
        return {
            "error": f"No Model is found.",
        }
    try:
        if query == "reactions":
            total = len(model.reactions)
            items = [f"{rxn.id} ({rxn.name})" for rxn in model.reactions][:count]
            return {"reactions": items, "shown": len(items), "total": total}
        elif query == "genes":
            total = len(model.genes)
            items = [f"{gn.id} ({gn.name})" for gn in model.genes][:count]
            return {"genes": items, "shown": len(items), "total": total}
        elif query == "metabolites":
            total = len(model.metabolites)
            items = [f"{mb.id} ({mb.name})" for mb in model.metabolites][:count]
            return {"metabolites": items, "shown": len(items), "total": total}
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
    if isinstance(model, dict):
        return {"error": "No model is currently loaded. Please load a model first."}

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
    obj = _clip_zero(solution.objective_value)
    model_manager.objective = obj
    return {
        "Objective value": str(obj),
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
    if isinstance(model, dict):
        return {"error": "No model is currently loaded. Please load a model first."}

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

    obj = _clip_zero(solution.objective_value)
    model_manager.objective = obj
    total_flux = float(solution.fluxes.abs().sum())

    return {
        "Objective value": str(obj),
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
    if isinstance(model, dict):
        return {"error": "No model is currently loaded. Please load a model first."}

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

    obj = _clip_zero(solution.objective_value)
    model_manager.objective = obj
    return {
        "Objective value": str(obj),
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
        fva_df["Maximum Flux"] = fva_df["Maximum Flux"].apply(_clip_zero)
        fva_df["Minimum Flux"] = fva_df["Minimum Flux"].apply(_clip_zero)

        if len(fva_df) > 5:
            output_dir = os.path.join(session_dir or ".", 'fva')
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
        if isinstance(model, dict):
            return {"error": "No model is currently loaded. Please load a model first."}
        _FUZZY_THRESHOLD = 0.8
        valid_genes = []
        not_found = []
        fuzzy_notes = []
        seen = set()
        for name in gene_names:
            tag, payload = _find_gene(model, name)
            if tag == "exact":
                gene = payload
            else:
                best = payload[0] if payload else None
                if best is None:
                    not_found.append({"gene": name, "suggestions": []})
                    continue
                q = name.strip().lower()
                score = max(_fuzzy_score(q, best.id.lower()), _fuzzy_score(q, best.name.lower()))
                if score < _FUZZY_THRESHOLD:
                    not_found.append({"gene": name, "suggestions": [g.id for g in payload]})
                    continue
                gene = best
                fuzzy_notes.append(f"'{name}' was fuzzy-matched to '{best.id}' (similarity {score:.0%})")
            if gene.id not in seen:
                valid_genes.append(gene)
                seen.add(gene.id)

        if not_found:
            return {
                "error": f"Gene(s) not found (no match above 80% similarity): {[e['gene'] for e in not_found]}",
                "suggestions": not_found,
            }
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
        result["Post-KO Growth"] = result["Post-KO Growth"].apply(_clip_zero)

        out: dict = {}
        if fuzzy_notes:
            out["fuzzy_matches"] = fuzzy_notes
        if len(result) > 5:
            file_path = os.path.join(session_dir or ".", "knockouts", "gene_knockout_result.csv")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            result.to_csv(file_path, index=False)
            subset = result.iloc[:5, :5]
            out.update({"file": file_path, "data": subset.to_dict(orient="records"), "note": "Too many results to display. Download CSV."})
            return out

        out["data"] = result.to_dict(orient="records")
        return out

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
        if isinstance(model, dict):
            return {"error": "No model is currently loaded. Please load a model first."}
        _FUZZY_THRESHOLD = 0.8
        valid_rxns = []
        not_found = []
        fuzzy_notes = []
        seen = set()
        for name in reaction_names:
            tag, payload = _find_reaction(model, name)
            if tag == "exact":
                rxn = payload
            else:
                best = payload[0] if payload else None
                if best is None:
                    not_found.append({"reaction": name, "suggestions": []})
                    continue
                q = name.strip().lower()
                score = max(_fuzzy_score(q, best.id.lower()), _fuzzy_score(q, best.name.lower()))
                if score < _FUZZY_THRESHOLD:
                    not_found.append({"reaction": name, "suggestions": [r.id for r in payload]})
                    continue
                rxn = best
                fuzzy_notes.append(f"'{name}' was fuzzy-matched to '{best.id}' (similarity {score:.0%})")
            if rxn.id not in seen:
                valid_rxns.append(rxn)
                seen.add(rxn.id)

        if not_found:
            return {
                "error": f"Reaction(s) not found (no match above 80% similarity): {[e['reaction'] for e in not_found]}",
                "suggestions": not_found,
            }
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
        result["Post-KO Growth"] = result["Post-KO Growth"].apply(_clip_zero)

        out: dict = {}
        if fuzzy_notes:
            out["fuzzy_matches"] = fuzzy_notes
        if len(result) > 5:
            file_path = os.path.join(session_dir or ".", "knockouts", "reaction_knockout_result.csv")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            result.to_csv(file_path, index=False)
            subset = result.iloc[:5, :5]
            out.update({"file": file_path, "data": subset.to_dict(orient="records"), "note": "Too many results to display. Download CSV."})
            return out

        out["data"] = result.to_dict(orient="records")
        return out

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
    try:
        model = model_manager.get_current_model()
    except Exception as e:
        return {"error": str(e)}
    if isinstance(model, dict):
        return {"error": "No model is currently loaded. Please load a model first."}

    try:
        config = recommend_sampling_config(model)
        config["method"] = model_manager.sampler

        if config["method"] == "achr":
            sampler = ACHRSampler(model, thinning=config["thinning"])
        else:
            sampler = OptGPSampler(model, thinning=config["thinning"], processes=config["processes"])
        samples = sampler.sample(reaction_count)
        subset = samples.iloc[:5, :5]
        sampling_dir = os.path.join(session_dir or ".", 'flux_sampling')
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
    except Exception as e:
        return {"error": str(e)}


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
            return {
                "status": "skipped",
                "reason": (
                    f"Cannot visualize fluxes — FBA returned '{solution.status}'. "
                    "Check that the model has valid bounds and a feasible growth condition before running Escher."
                ),
            }

        reaction_data = solution.fluxes.to_dict()

        # Save HTML to session output directory
        escher_dir = os.path.join(session_dir or ".", "escher")
        os.makedirs(escher_dir, exist_ok=True)
        safe_name = map_name.replace("/", "_").replace(" ", "_")
        filepath = os.path.join(escher_dir, f"{safe_name}_flux.html")

        builder = escher.Builder(map_name=map_name, model=model, reaction_data=reaction_data)
        builder.save_html(filepath)

        obj = _clip_zero(solution.objective_value)
        return {
            "response": f"Escher flux map saved to {filepath}. Map: '{map_name}'. Objective value: {obj:.4f}",
            "file": filepath,
            "map_name": map_name,
            "objective_value": obj
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
            raw_msg = case.get("message")
            message = str(raw_msg) if raw_msg else ""

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

        # Save full HTML report and extract total score.
        # snapshot_report calls compute_score() internally, which writes
        # result["score"]["total_score"] and result["score"]["sections"] in place.
        report_path = None
        total_score_pct = None
        section_scores = {}
        try:
            html = snapshot_report(result)
            report_dir = os.path.join(session_dir or ".", "memote")
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, "memote_report.html")
            with open(report_path, "w", encoding="utf-8") as fh:
                fh.write(html)
            score_block = result.get("score", {})
            raw_total = score_block.get("total_score")
            if raw_total is not None:
                total_score_pct = round(raw_total * 100, 1)
            for sec in score_block.get("sections", []):
                section_scores[sec["section"]] = round(sec["score"] * 100, 1)
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
        if total_score_pct is not None:
            base["total_score"] = f"{total_score_pct}%"
        if section_scores:
            base["scores_by_section"] = {k: f"{v}%" for k, v in section_scores.items()}
        if report_path:
            base["full_report"] = report_path

        score_str = f" Total memote score: {total_score_pct}%." if total_score_pct is not None else ""

        if not failed:
            return {
                **base,
                "status": "passed",
                "summary": f"All {len(passed)} memote tests passed.{score_str}",
            }

        TOO_MANY = 10
        if len(failed) > TOO_MANY:
            return {
                **base,
                "status": "failed",
                "summary": (
                    f"{len(failed)} out of {total_run} tests failed.{score_str} "
                    f"Showing top {min(5, len(failed))} failures below."
                ),
                "major_failures": [_fmt(f) for f in failed[:5]],
            }
        else:
            return {
                **base,
                "status": "failed",
                "summary": f"{len(failed)} out of {total_run} tests failed.{score_str}",
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
    if isinstance(model, dict):
        return {"error": "No model is currently loaded. Please load a model first."}

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
    CSV batch mode        — provide csv_path to a CSV with 3 columns: reaction_id, lower_bound, upper_bound.
                           Column names are ignored; order is positional. Header row is auto-detected.
    """
    import pandas as _pd, os as _os

    try:
        model = model_manager.get_current_model()
    except Exception as e:
        return {"error": str(e)}
    if isinstance(model, dict):
        return {"error": "No model is currently loaded. Please load a model first."}

    # ── CSV batch mode ──────────────────────────────────────────────────────────
    if csv_path is not None:
        if not _os.path.isabs(csv_path) and session_uploads_dir:
            csv_path = _os.path.join(session_uploads_dir, csv_path)
        if not _os.path.exists(csv_path):
            return {"error": f"CSV file not found: {csv_path}"}
        try:
            # Read without assuming headers; positionally map col0=rxn_id, col1=lb, col2=ub.
            df = _pd.read_csv(csv_path, header=None)
            if df.shape[1] < 3:
                return {"error": f"CSV must have at least 3 columns (reaction_id, lb, ub). Found {df.shape[1]}."}
            # Auto-detect a header row: if col1 or col2 of the first row can't be cast to float, skip it.
            try:
                float(df.iloc[0, 1])
                float(df.iloc[0, 2])
            except (ValueError, TypeError):
                df = df.iloc[1:].reset_index(drop=True)
            df = df.iloc[:, :3]
            df.columns = ["rxn_id", "lb", "ub"]
        except Exception as e:
            return {"error": f"Could not read CSV: {e}"}

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



def build_model_with_carveme(
    fasta_file: str,
    model_id: str = None,
    gram: str = None,
) -> dict:
    import subprocess as _sp

    if not os.path.isabs(fasta_file) and session_uploads_dir:
        fasta_file = os.path.join(session_uploads_dir, fasta_file)
    if not os.path.exists(fasta_file):
        return {"error": f"FASTA file not found: {fasta_file}"}

    output_dir = os.path.join(session_dir, "carveme") if session_dir else "carveme"
    os.makedirs(output_dir, exist_ok=True)

    if not model_id:
        model_id = os.path.splitext(os.path.basename(fasta_file))[0]

    out_path = os.path.join(output_dir, f"{model_id}.xml")

    def _is_dna(fasta_path: str) -> bool:
        with open(fasta_path, "r") as fh:
            seq = "".join(
                line.strip() for line in fh if not line.startswith(">")
            )[:200]
        return bool(seq) and all(c in "ACGTNacgtn" for c in seq)

    protein_fasta = fasta_file
    prodigal_used = False

    if _is_dna(fasta_file):
        protein_fasta = os.path.join(output_dir, f"{model_id}_proteins.faa")
        prodigal_cmd = [
            "prodigal",
            "-i", fasta_file,
            "-a", protein_fasta,
            "-p", "meta",
        ]
        try:
            _sp.run(prodigal_cmd, check=True, capture_output=True)
            prodigal_used = True
        except FileNotFoundError:
            return {
                "error": "Prodigal executable not found.",
                "fix": "Install with: sudo apt install prodigal  OR  conda install -c bioconda prodigal",
            }
        except _sp.CalledProcessError as e:
            return {
                "error": "Prodigal failed.",
                "detail": e.stderr.decode(errors="replace") if e.stderr else str(e),
            }

    carve_cmd = ["carve", protein_fasta, "-o", out_path]
    if gram in ("grampos", "gramneg"):
        carve_cmd += ["--gram", gram]

    try:
        result = _sp.run(carve_cmd, capture_output=True)
    except FileNotFoundError:
        return {"error": "CarveMe (carve) not found. Install with: pip install carveme"}

    if result.returncode != 0 or not os.path.exists(out_path):
        stderr = result.stderr.decode(errors="replace").strip() if result.stderr else ""
        stdout = result.stdout.decode(errors="replace").strip() if result.stdout else ""
        detail = stderr or stdout or "No output captured."
        return {"error": "CarveMe failed to produce an output model.", "detail": detail}

    try:
        loaded_id = model_manager.load_sbml(out_path)
        model = model_manager.get_current_model()
        return {
            "status": "success",
            "model_id": loaded_id,
            "output_file": out_path,
            "prodigal_used": prodigal_used,
            "reactions": len(model.reactions),
            "metabolites": len(model.metabolites),
            "genes": len(model.genes),
        }
    except Exception as e:
        return {
            "status": "success",
            "warning": f"Model saved but could not be auto-loaded: {e}",
            "output_file": out_path,
            "prodigal_used": prodigal_used,
        }


def _detect_columns(df):
    """Pick the id column and value column from a confidence/expression CSV."""
    cols = {c.lower().strip(): c for c in df.columns}

    id_candidates = ["gene", "gene_id", "genes", "gene_name", "locus",
                     "reaction", "rxn", "rxn_id", "reaction_id", "id"]
    val_candidates = ["confidence", "conf", "score", "expression", "expr",
                      "value", "level", "tpm", "fpkm", "rpkm", "counts", "count"]

    id_col = next((cols[c] for c in id_candidates if c in cols), None)
    val_col = next((cols[c] for c in val_candidates if c in cols), None)

    # Fall back to positional: first column = ids, second = values
    if id_col is None:
        id_col = df.columns[0]
    if val_col is None:
        remaining = [c for c in df.columns if c != id_col]
        val_col = remaining[0] if remaining else None
    return id_col, val_col


def _expression_to_confidence(values, high_pct, mid_pct, low_pct):
    """Bin continuous expression values into CORDA confidence classes.

    >= high_pct percentile -> 3 (high)
    >= mid_pct  percentile -> 2 (medium)
    >= low_pct  percentile -> 1 (low)
    <  low_pct  percentile -> -1 (treated as not expressed)
    """
    import numpy as _np
    arr = _np.asarray(list(values), dtype=float)
    hi, mid, lo = _np.percentile(arr, [high_pct, mid_pct, low_pct])

    def _bin(v):
        if v >= hi:
            return 3
        if v >= mid:
            return 2
        if v >= lo:
            return 1
        return -1

    return [_bin(v) for v in arr], {"low": float(lo), "medium": float(mid), "high": float(hi)}


def _strip_version(token: str) -> str:
    """Drop a trailing version suffix, e.g. 'ENSG00000139618.15' -> 'ENSG00000139618'."""
    import re as _re
    return _re.sub(r"\.\d+$", "", token)


def _build_gene_index(model) -> dict:
    """Map every identifier the model knows for each gene -> model gene id.

    Pulls from the gene id, the gene name/symbol, and every cross-reference in
    gene.annotation (ncbigene, ensembl, uniprot, refseq, ...). Lets an uploaded
    expression/confidence file match the model regardless of which namespace it
    uses. The model gene id always wins over weaker (e.g. symbol) matches.
    """
    index: dict = {}

    def _add(key, gid, overwrite=False):
        if key is None:
            return
        k = str(key).strip().lower()
        if not k:
            return
        if overwrite or k not in index:
            index[k] = gid
        base = _strip_version(k)
        if base != k and base not in index:
            index[base] = gid

    # Lowest priority first (symbol), then annotation cross-refs, then id (wins).
    for g in model.genes:
        if g.name:
            _add(g.name, g.id)
    for g in model.genes:
        for v in (g.annotation or {}).values():
            if isinstance(v, (list, tuple, set)):
                for item in v:
                    _add(item, g.id)
            else:
                _add(v, g.id)
    for g in model.genes:
        _add(g.id, g.id, overwrite=True)
    return index


def _match_gene(raw_id: str, index: dict):
    """Resolve an uploaded id to a model gene id via the cross-reference index."""
    k = str(raw_id).strip().lower()
    if k in index:
        return index[k]
    base = _strip_version(k)
    return index.get(base)


def build_context_model_with_corda(
    data_csv: str,
    data_type: str = "auto",
    model_id: str = None,
    high_percentile: float = 75.0,
    mid_percentile: float = 50.0,
    low_percentile: float = 25.0,
    keep_objective_high: bool = True,
    met_prod: str = None,
) -> dict:
    """
    Build a context-specific genome-scale model from the currently loaded base
    (reference) model using CORDA (Cost Optimization Reaction Dependency Assessment).

    The uploaded CSV is auto-detected as one of:
      * gene expression      — 2 columns: gene id + a continuous expression value
                               (TPM/FPKM/counts). Values are binned into CORDA
                               confidence classes (-1,1,2,3) by percentile, so the
                               user only needs to provide raw expression data.
      * gene confidence      — 2 columns: gene id + an integer confidence in
                               {-1,0,1,2,3}.
      * reaction confidence  — 2 columns: reaction id + an integer confidence in
                               {-1,0,1,2,3}.

    data_type forces the interpretation: 'auto' (default), 'expression',
    'gene_confidence', or 'reaction_confidence'.

    Gene ids in the file do NOT have to match the model's own gene ids. They are
    resolved through a cross-reference index built from the model (gene id, gene
    symbol/name, and every annotation xref — Ensembl, Entrez/ncbigene, UniProt,
    RefSeq, ...), with version suffixes like '.15' stripped. This means a file
    downloaded from GTEx, Expression Atlas, GEO, etc. can often be used directly.
    The returned 'id_mapping_coverage' reports how many model genes were matched.
    """
    import numpy as _np
    import pandas as _pd

    try:
        model = model_manager.get_current_model()
        if isinstance(model, dict):  # ModelManager returns a dict when nothing loaded
            return {"error": "No model is currently loaded. Load a base/reference model first."}
    except Exception as e:
        return {"error": str(e)}

    # Resolve CSV path (relative -> session uploads dir)
    if not os.path.isabs(data_csv) and session_uploads_dir:
        data_csv = os.path.join(session_uploads_dir, data_csv)
    if not os.path.exists(data_csv):
        return {"error": f"Data CSV not found: {data_csv}"}

    try:
        df = _pd.read_csv(data_csv)
    except Exception as e:
        return {"error": f"Could not read CSV: {e}"}
    if df.shape[1] < 2:
        return {"error": "CSV must have at least two columns: an id column and a value column."}

    id_col, val_col = _detect_columns(df)
    if val_col is None:
        return {"error": "Could not identify a value column in the CSV."}

    ids = df[id_col].astype(str).str.strip().tolist()
    raw_vals = _pd.to_numeric(df[val_col], errors="coerce")
    if raw_vals.isna().all():
        return {"error": f"Value column '{val_col}' has no numeric values."}

    # ── Decide id level: gene-level vs reaction-level ──────────────────────────
    # Genes are matched through a model-derived cross-reference index, so the
    # uploaded file may key on Ensembl, Entrez, UniProt, RefSeq or gene symbols
    # (with or without version suffixes) — not just the model's own gene ids.
    gene_index = _build_gene_index(model)
    rxn_ids = {r.id for r in model.reactions}

    gene_hits = [_match_gene(i, gene_index) for i in ids]
    n_gene_match = sum(1 for gid in gene_hits if gid is not None)
    n_rxn_match = sum(1 for i in ids if i in rxn_ids)

    # ── Decide value semantics: discrete confidence vs continuous expression ───
    nonnull = raw_vals.dropna()
    looks_like_confidence = bool(nonnull.isin([-1, 0, 1, 2, 3]).all())

    if data_type == "reaction_confidence":
        level, semantics = "reaction", "confidence"
    elif data_type == "gene_confidence":
        level, semantics = "gene", "confidence"
    elif data_type == "expression":
        level, semantics = ("reaction" if n_rxn_match > n_gene_match else "gene"), "expression"
    else:  # auto
        level = "reaction" if n_rxn_match > n_gene_match else "gene"
        semantics = "confidence" if looks_like_confidence else "expression"

    if level == "gene" and n_gene_match == 0:
        return {"error": (
            f"None of the {len(ids)} ids in column '{id_col}' could be matched to genes in "
            f"the loaded model '{model_manager.current_model_id}' — even after checking gene "
            "ids, symbols and annotation cross-references (Ensembl/Entrez/UniProt/RefSeq). "
            "The file is likely for a different organism, or the model carries no gene "
            "annotations to map against (e.g. the e_coli_core textbook model)."
        )}
    if level == "reaction" and n_rxn_match == 0:
        return {"error": (
            f"None of the ids in column '{id_col}' match reaction IDs in the loaded model. "
            "Check the identifiers, or set data_type explicitly."
        )}

    # ── Resolve uploaded ids -> model entity, aggregating duplicates ───────────
    # Several uploaded rows can map to the same model gene (multiple probes or
    # transcripts); collapse them by taking the max value, the usual convention.
    mapped_value: dict = {}
    matched_rows = 0
    if level == "gene":
        for gid, v in zip(gene_hits, raw_vals):
            if gid is None or _pd.isna(v):
                continue
            matched_rows += 1
            mapped_value[gid] = max(float(v), mapped_value.get(gid, float("-inf")))
        total = len(model.genes)
        coverage = {
            "model_genes_total": total,
            "model_genes_matched": len(mapped_value),
            "model_genes_match_pct": round(100.0 * len(mapped_value) / total, 1) if total else 0.0,
            "uploaded_rows": len(ids),
            "uploaded_rows_matched": matched_rows,
            "uploaded_rows_unmatched": len(ids) - matched_rows,
        }
    else:  # reaction level: ids are matched directly against reaction ids
        for i, v in zip(ids, raw_vals):
            if i not in rxn_ids or _pd.isna(v):
                continue
            matched_rows += 1
            mapped_value[i] = max(float(v), mapped_value.get(i, float("-inf")))
        total = len(model.reactions)
        coverage = {
            "model_reactions_total": total,
            "model_reactions_matched": len(mapped_value),
            "uploaded_rows": len(ids),
            "uploaded_rows_matched": matched_rows,
            "uploaded_rows_unmatched": len(ids) - matched_rows,
        }

    # ── Convert aggregated values to a confidence map keyed by model entity id ─
    binning = None
    entity_conf: dict = {}
    if semantics == "expression":
        keys = list(mapped_value.keys())
        conf_list, binning = _expression_to_confidence(
            [mapped_value[k] for k in keys], high_percentile, mid_percentile, low_percentile
        )
        entity_conf = dict(zip(keys, conf_list))
    else:
        for k, v in mapped_value.items():
            iv = int(round(v))
            if iv not in (-1, 0, 1, 2, 3):
                return {"error": f"Confidence value {v} for '{k}' is invalid. Allowed: -1,0,1,2,3."}
            entity_conf[k] = iv

    # ── Build the reaction confidence dict CORDA needs ─────────────────────────
    np_shim = not hasattr(_np, "in1d")
    if np_shim:  # corda 0.5.x still calls the numpy-1.x alias removed in numpy 2.0
        _np.in1d = _np.isin
    try:
        from corda import reaction_confidence, CORDA
    except ImportError:
        return {"error": "The 'corda' package is not installed. Run: pip install corda"}

    if level == "reaction":
        rxn_conf = {r.id: entity_conf.get(r.id, 0) for r in model.reactions}
    else:
        rxn_conf = {}
        for r in model.reactions:
            if r.gene_reaction_rule.strip():
                rxn_conf[r.id] = reaction_confidence(r, entity_conf)
            else:
                rxn_conf[r.id] = 0

    # Keep the objective reaction(s) high-confidence so the context model can grow
    objective_rxns = []
    if keep_objective_high and model.objective and model.objective.expression:
        for var in model.objective.expression.free_symbols:
            rid = var.name.replace("_reverse", "").split(" ")[0]
            # objective variable names map onto reaction ids
            base = rid
            if model.reactions.has_id(base):
                rxn_conf[base] = 3
                objective_rxns.append(base)

    conf_hist = {c: sum(1 for v in rxn_conf.values() if v == c) for c in (-1, 0, 1, 2, 3)}

    if not model_id:
        model_id = f"{model_manager.current_model_id}_corda"

    # ── Run CORDA ──────────────────────────────────────────────────────────────
    try:
        met_prod_arg = None
        if met_prod:
            met_prod_arg = [m.strip() for m in met_prod.split(",") if m.strip()]
        opt = CORDA(model, rxn_conf, met_prod=met_prod_arg) if met_prod_arg else CORDA(model, rxn_conf)
        opt.build()
        rec_model = opt.cobra_model(model_id)
    except Exception as e:
        return {"error": f"CORDA reconstruction failed: {e}"}

    # ── Save SBML and load into the session ────────────────────────────────────
    output_dir = os.path.join(session_dir, "corda") if session_dir else "corda"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_id}.xml")
    try:
        write_sbml_model(rec_model, out_path)
    except Exception as e:
        return {"error": f"CORDA model built but could not be saved as SBML: {e}"}

    included = sum(1 for v in opt.included.values() if v)
    result = {
        "status": "success",
        "model_id": model_id,
        "output_file": out_path,
        "interpreted_as": f"{level}-level {semantics}",
        "id_column": id_col,
        "value_column": val_col,
        "id_mapping_coverage": coverage,
        "base_reactions": len(model.reactions),
        "reconstructed_reactions": len(rec_model.reactions),
        "reconstructed_metabolites": len(rec_model.metabolites),
        "reconstructed_genes": len(rec_model.genes),
        "reactions_included_by_corda": included,
        "confidence_distribution": conf_hist,
    }
    if binning:
        result["expression_percentile_thresholds"] = {
            "low_percentile": low_percentile, "mid_percentile": mid_percentile,
            "high_percentile": high_percentile, "value_cutoffs": binning,
        }
    if objective_rxns:
        result["objective_kept_high_confidence"] = objective_rxns
    if level == "gene" and coverage.get("model_genes_match_pct", 100) < 10:
        result["warning"] = (
            f"Only {coverage['model_genes_matched']}/{coverage['model_genes_total']} "
            f"({coverage['model_genes_match_pct']}%) of model genes were matched by the "
            "uploaded file. The reconstruction may be poorly constrained — check that the "
            "file is for the right organism and uses identifiers the model annotates."
        )

    try:
        loaded_id = model_manager.load_sbml(out_path)
        result["loaded_model_id"] = loaded_id
    except Exception as e:
        result["warning"] = f"Model saved but could not be auto-loaded: {e}"
    return result


def request_file_upload(param: str, file_types: str, description: str) -> dict:
    """
    Call this when a required file has not been uploaded yet and you cannot proceed without it.
    param: exact parameter name the file maps to (e.g. 'data_csv', 'fasta_file', 'csv_path').
    file_types: comma-separated accepted extensions (e.g. 'csv,tsv' or 'faa,fasta,fa,fna').
    description: one sentence telling the user what the file must contain.
    After calling this, STOP — do not call any other tool in the same turn.
    """
    return {"__upload_required__": True, "param": param, "file_types": file_types, "description": description}


def request_tool_inputs(tool_name: str, prefilled_params: str, explanation: str) -> dict:
    """
    Call this when you know which tool to run but the user has not supplied all required arguments
    and they cannot be safely inferred from the message.
    tool_name: exact name of the tool you intend to call (e.g. 'add_reaction').
    prefilled_params: JSON string of argument values you CAN derive from the user's message,
        e.g. '{"reaction_id": "GALK"}'. Use '{}' if nothing is derivable.
    explanation: one sentence describing what you understood from the user's request.
    After calling this, STOP — do not call any other tool in the same turn.
    The system will present the user with a form; once submitted, the tool runs automatically.
    """
    return {
        "__tool_inputs_required__": True,
        "tool_name": tool_name,
        "prefilled_params": prefilled_params,
        "explanation": explanation,
    }


def _apply_uploaded_bounds(model):
    """Apply any reaction bounds uploaded via CSV (model_manager.bounds_data) in place.

    Mirrors the bounds handling in run_fba/run_pfba so growth-dependent analyses
    (essentiality, minimal medium) respect the same nutrient/flux constraints.
    """
    bounds = model_manager.bounds_data
    if bounds:
        for rxn_id, lval, uval in bounds:
            if model.reactions.has_id(rxn_id):
                model.reactions.get_by_id(rxn_id).bounds = (lval, uval)


def find_essential_genes(threshold: float = None) -> dict:
    """
    Find the essential genes of the currently loaded model.

    Core cobrapy function: cobra.flux_analysis.find_essential_genes.
    A gene is essential if knocking out all reactions that depend on it drops
    the objective (growth) to zero / below the threshold / infeasible.
    """
    from cobra.flux_analysis import find_essential_genes as _feg
    try:
        model = model_manager.get_current_model()
        if isinstance(model, dict):
            return {"error": "No model is currently loaded. Please load a model first."}
    except Exception as e:
        return {"error": str(e)}

    if not model.objective.expression:
        return {"error": "No objective function is set on the model. Set an objective before finding essential genes."}

    _apply_uploaded_bounds(model)

    try:
        genes = _feg(model, threshold=threshold)
    except Exception as e:
        return {"error": str(e)}

    rows = sorted(({"id": g.id, "name": g.name} for g in genes), key=lambda d: d["id"])

    if len(rows) > 5:
        import pandas as _pd
        out_dir = os.path.join(session_dir or ".", "essentiality")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "essential_genes.csv")
        _pd.DataFrame(rows).to_csv(csv_path, index=False)
        return {
            "essential_gene_count": len(rows),
            "total_genes": len(model.genes),
            "threshold": threshold if threshold is not None else "1% of max objective (default)",
            "first_5": rows[:5],
            "message": f"{len(rows)} essential genes found. Full list saved to {csv_path}.",
        }
    return {
        "essential_gene_count": len(rows),
        "total_genes": len(model.genes),
        "threshold": threshold if threshold is not None else "1% of max objective (default)",
        "essential_genes": rows,
    }


def find_essential_reactions(threshold: float = None) -> dict:
    """
    Find the essential reactions of the currently loaded model.

    Core cobrapy function: cobra.flux_analysis.find_essential_reactions.
    A reaction is essential if constraining its flux to zero drops the objective
    (growth) to zero / below the threshold / infeasible.
    """
    from cobra.flux_analysis import find_essential_reactions as _fer
    try:
        model = model_manager.get_current_model()
        if isinstance(model, dict):
            return {"error": "No model is currently loaded. Please load a model first."}
    except Exception as e:
        return {"error": str(e)}

    if not model.objective.expression:
        return {"error": "No objective function is set on the model. Set an objective before finding essential reactions."}

    _apply_uploaded_bounds(model)

    try:
        reactions = _fer(model, threshold=threshold)
    except Exception as e:
        return {"error": str(e)}

    rows = sorted(({"id": r.id, "name": r.name} for r in reactions), key=lambda d: d["id"])

    if len(rows) > 5:
        import pandas as _pd
        out_dir = os.path.join(session_dir or ".", "essentiality")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "essential_reactions.csv")
        _pd.DataFrame(rows).to_csv(csv_path, index=False)
        return {
            "essential_reaction_count": len(rows),
            "total_reactions": len(model.reactions),
            "threshold": threshold if threshold is not None else "1% of max objective (default)",
            "first_5": rows[:5],
            "message": f"{len(rows)} essential reactions found. Full list saved to {csv_path}.",
        }
    return {
        "essential_reaction_count": len(rows),
        "total_reactions": len(model.reactions),
        "threshold": threshold if threshold is not None else "1% of max objective (default)",
        "essential_reactions": rows,
    }


def check_model_consistency(flux_threshold: float = 1.0) -> dict:
    """
    Check the consistency of the loaded model using FASTCC and remove blocked reactions.

    Core cobrapy function: cobra.flux_analysis.fastcc.
    FASTCC (Fast Consistency Check) returns a consistent model with all blocked
    (flux-inconsistent) reactions removed. The cleaned, consistent model REPLACES
    the currently loaded model in the session.
    """
    from cobra.flux_analysis import fastcc as _fastcc
    try:
        model = model_manager.get_current_model()
        if isinstance(model, dict):
            return {"error": "No model is currently loaded. Please load a model first."}
    except Exception as e:
        return {"error": str(e)}

    _apply_uploaded_bounds(model)

    try:
        consistent = _fastcc(model, flux_threshold=flux_threshold)
    except Exception as e:
        return {"error": str(e)}

    original_rxn_ids = {r.id for r in model.reactions}
    consistent_rxn_ids = {r.id for r in consistent.reactions}
    blocked = sorted(original_rxn_ids - consistent_rxn_ids)

    # Replace the active model with the consistent one (keep the same model id).
    current_id = model_manager.current_model_id
    model_manager.models[current_id] = consistent

    result = {
        "status": "success",
        "model_id": current_id,
        "flux_threshold": flux_threshold,
        "reactions_before": len(original_rxn_ids),
        "reactions_after": len(consistent_rxn_ids),
        "blocked_reactions_removed": len(blocked),
        "note": "The loaded model has been replaced with the consistent (blocked-reaction-free) model.",
    }
    if blocked:
        if len(blocked) > 20:
            result["blocked_reactions_sample"] = blocked[:20]
            result["message"] = f"{len(blocked)} blocked reactions removed (showing first 20)."
        else:
            result["blocked_reactions"] = blocked
    return result


def check_mass_balance() -> dict:
    """
    Check the mass and charge balance of every reaction in the loaded model.

    Core cobrapy function: cobra.manipulation.check_mass_balance.
    Returns the unbalanced reactions together with the element/charge imbalance.
    An empty result means all reactions are balanced.
    """
    from cobra.manipulation import check_mass_balance as _cmb
    try:
        model = model_manager.get_current_model()
        if isinstance(model, dict):
            return {"error": "No model is currently loaded. Please load a model first."}
    except Exception as e:
        return {"error": str(e)}

    try:
        imbalance = _cmb(model)
    except Exception as e:
        return {"error": str(e)}

    rows = [
        {
            "reaction_id": rxn.id,
            "reaction_name": rxn.name,
            "imbalance": {str(k): float(v) for k, v in elements.items()},
        }
        for rxn, elements in imbalance.items()
    ]

    if not rows:
        return {
            "status": "balanced",
            "unbalanced_reaction_count": 0,
            "total_reactions": len(model.reactions),
            "summary": "All reactions are mass- and charge-balanced.",
        }

    if len(rows) > 5:
        import pandas as _pd
        out_dir = os.path.join(session_dir or ".", "validation")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "mass_imbalance.csv")
        _pd.DataFrame(
            [{"reaction_id": r["reaction_id"], "reaction_name": r["reaction_name"],
              "imbalance": str(r["imbalance"])} for r in rows]
        ).to_csv(csv_path, index=False)
        return {
            "status": "unbalanced",
            "unbalanced_reaction_count": len(rows),
            "total_reactions": len(model.reactions),
            "first_5": rows[:5],
            "message": f"{len(rows)} unbalanced reactions found. Full list saved to {csv_path}.",
        }
    return {
        "status": "unbalanced",
        "unbalanced_reaction_count": len(rows),
        "total_reactions": len(model.reactions),
        "unbalanced_reactions": rows,
    }


def prune_unused_reactions() -> dict:
    """
    Remove reactions that have no assigned metabolites from the loaded model.

    Core cobrapy function: cobra.manipulation.prune_unused_reactions.
    The pruned model REPLACES the currently loaded model in the session.
    """
    from cobra.manipulation import prune_unused_reactions as _prune
    try:
        model = model_manager.get_current_model()
        if isinstance(model, dict):
            return {"error": "No model is currently loaded. Please load a model first."}
    except Exception as e:
        return {"error": str(e)}

    before = len(model.reactions)
    try:
        pruned_model, removed = _prune(model)
    except Exception as e:
        return {"error": str(e)}

    current_id = model_manager.current_model_id
    model_manager.models[current_id] = pruned_model

    removed_ids = [r.id for r in removed]
    return {
        "status": "success",
        "model_id": current_id,
        "reactions_before": before,
        "reactions_after": len(pruned_model.reactions),
        "removed_count": len(removed_ids),
        "removed_reactions": removed_ids[:50],
        "note": "The loaded model has been replaced with the pruned model."
                + (" Showing first 50 removed reaction IDs." if len(removed_ids) > 50 else ""),
    }


def prune_unused_metabolites() -> dict:
    """
    Remove metabolites that are not involved in any reaction from the loaded model.

    Core cobrapy function: cobra.manipulation.prune_unused_metabolites.
    The pruned model REPLACES the currently loaded model in the session.
    """
    from cobra.manipulation import prune_unused_metabolites as _prune
    try:
        model = model_manager.get_current_model()
        if isinstance(model, dict):
            return {"error": "No model is currently loaded. Please load a model first."}
    except Exception as e:
        return {"error": str(e)}

    before = len(model.metabolites)
    try:
        pruned_model, removed = _prune(model)
    except Exception as e:
        return {"error": str(e)}

    current_id = model_manager.current_model_id
    model_manager.models[current_id] = pruned_model

    removed_ids = [m.id for m in removed]
    return {
        "status": "success",
        "model_id": current_id,
        "metabolites_before": before,
        "metabolites_after": len(pruned_model.metabolites),
        "removed_count": len(removed_ids),
        "removed_metabolites": removed_ids[:50],
        "note": "The loaded model has been replaced with the pruned model."
                + (" Showing first 50 removed metabolite IDs." if len(removed_ids) > 50 else ""),
    }


def find_minimal_medium(
    min_objective_value: float = 0.1,
    minimize_components: bool = False,
    open_exchanges: bool = False,
) -> dict:
    """
    Find the minimal growth medium for the loaded model.

    Core cobrapy function: cobra.medium.minimal_medium.
    Returns the import flux for each exchange reaction required to sustain at
    least `min_objective_value` growth. Set `minimize_components=True` to instead
    minimise the NUMBER of medium ingredients (slower, mixed-integer problem).
    """
    from cobra.medium import minimal_medium as _mm
    import pandas as _pd
    try:
        model = model_manager.get_current_model()
        if isinstance(model, dict):
            return {"error": "No model is currently loaded. Please load a model first."}
    except Exception as e:
        return {"error": str(e)}

    if not model.objective.expression:
        return {"error": "No objective function is set on the model. Set an objective (e.g. biomass) before finding a minimal medium."}

    _apply_uploaded_bounds(model)

    try:
        medium = _mm(
            model,
            min_objective_value=min_objective_value,
            minimize_components=minimize_components,
            open_exchanges=open_exchanges,
        )
    except Exception as e:
        return {"error": str(e)}

    if medium is None:
        return {
            "error": (
                "Minimal medium computation was infeasible — the requested "
                f"min_objective_value ({min_objective_value}) likely exceeds the maximum "
                "achievable growth rate. Try a lower value or open_exchanges=True."
            )
        }

    series = medium if isinstance(medium, _pd.Series) else medium.iloc[:, 0]
    rows = [{"exchange_reaction": rid, "import_flux": float(val)} for rid, val in series.items()]
    rows.sort(key=lambda d: d["exchange_reaction"])

    return {
        "status": "success",
        "min_objective_value": min_objective_value,
        "minimize_components": minimize_components,
        "open_exchanges": open_exchanges,
        "component_count": len(rows),
        "minimal_medium": rows,
    }


def build_model_with_mackinac(
    genome_id: str,
    username: str,
    password: str,
    source: str = "patric",
    model_id: str = None,
) -> dict:
    """
    Reconstruct a genome-scale metabolic model from a PATRIC genome using ModelSEED,
    via Mackinac, and load it as a COBRA model.

    Core functions (mackinac package):
      mackinac.get_token(username, password)                  — authenticate to PATRIC/ModelSEED
      mackinac.reconstruct_modelseed_model(genome_id, source) — build & gap-fill a draft model
      mackinac.create_cobra_model_from_modelseed_model(id)    — convert to a COBRApy model

    Requires PATRIC account credentials and live internet access to the ModelSEED
    web service. The reconstructed model is saved as SBML and loaded into the session.
    """
    try:
        import mackinac
    except ImportError:
        return {"error": "The 'mackinac' package is not installed. Run: pip install mackinac"}

    # Point mackinac at the current ModelSEED endpoint.
    mackinac.modelseed.ms_client.url = "https://modelseed.org/api/model"

    # 1. Authenticate to the PATRIC/ModelSEED web service.
    try:
        mackinac.get_token(username, password)
    except Exception as e:
        return {"error": f"Authentication to PATRIC/ModelSEED failed: {e}"}

    # 2. Reconstruct (and gap-fill) a draft ModelSEED model from the genome.
    try:
        stats = mackinac.reconstruct_modelseed_model(genome_id, source=source)
    except Exception as e:
        return {"error": f"ModelSEED reconstruction failed for genome '{genome_id}': {e}"}

    seed_model_id = None
    if isinstance(stats, dict):
        seed_model_id = stats.get("id") or stats.get("ref")
    if not seed_model_id:
        seed_model_id = genome_id

    # 3. Convert the ModelSEED model into a COBRApy model.
    try:
        cobra_model = mackinac.create_cobra_model_from_modelseed_model(seed_model_id)
    except Exception as e:
        return {"error": f"Could not convert ModelSEED model '{seed_model_id}' to a COBRA model: {e}"}

    if not model_id:
        model_id = f"{genome_id}_modelseed"
    cobra_model.id = model_id

    output_dir = os.path.join(session_dir, "mackinac") if session_dir else "mackinac"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_id}.xml")
    try:
        write_sbml_model(cobra_model, out_path)
    except Exception as e:
        return {"error": f"Model reconstructed but could not be saved as SBML: {e}"}

    model_manager.models[model_id] = cobra_model
    model_manager.current_model_id = model_id
    if cobra_model.objective:
        model_manager.objective = True

    return {
        "status": "success",
        "model_id": model_id,
        "genome_id": genome_id,
        "source": source,
        "output_file": out_path,
        "reactions": len(cobra_model.reactions),
        "metabolites": len(cobra_model.metabolites),
        "genes": len(cobra_model.genes),
    }


def _objective_reaction_coeffs(model):
    """Return {reaction: coefficient} for the model's linear objective."""
    from cobra.util.solver import linear_reaction_coefficients
    return linear_reaction_coefficients(model)


def _resolve_knockouts(model, knockout_genes, knockout_reactions):
    """Resolve gene/reaction knockout targets (str or list) to model objects.

    Returns (gene_objs, rxn_objs, errors). Uses the same fuzzy matching as the
    rest of tools.py so users can pass ids or names.
    """
    def _as_list(x):
        if x is None:
            return []
        if isinstance(x, str):
            return [t.strip() for t in x.split(",") if t.strip()]
        return list(x)

    gene_objs, rxn_objs, errors = [], [], []
    seen_g, seen_r = set(), set()

    for name in _as_list(knockout_genes):
        tag, payload = _find_gene(model, name)
        g = payload if tag == "exact" else (payload[0] if payload else None)
        if g is None:
            errors.append({"gene": name, "reason": "not found"})
        elif g.id not in seen_g:
            gene_objs.append(g)
            seen_g.add(g.id)

    for name in _as_list(knockout_reactions):
        tag, payload = _find_reaction(model, name)
        r = payload if tag == "exact" else (payload[0] if payload else None)
        if r is None:
            errors.append({"reaction": name, "reason": "not found"})
        elif r.id not in seen_r:
            rxn_objs.append(r)
            seen_r.add(r.id)

    return gene_objs, rxn_objs, errors


def _flux_change_report(reference, perturbed, label, top_n=10):
    """Summarise the largest flux changes between two cobra Solutions.

    Returns a dict with the top_n reactions by absolute flux change and, if the
    full table is large, writes it to a CSV under the session directory.
    """
    import pandas as _pd
    diff = (perturbed.fluxes - reference.fluxes)
    table = _pd.DataFrame({
        "reaction_id": diff.index,
        "wild_type_flux": reference.fluxes.values,
        "perturbed_flux": perturbed.fluxes.values,
        "flux_change": diff.values,
    })
    table["abs_change"] = table["flux_change"].abs()
    table = table.sort_values("abs_change", ascending=False).drop(columns="abs_change")

    out = {"top_flux_changes": table.head(top_n).round(6).to_dict(orient="records")}
    changed = table[table["flux_change"].abs() > 1e-6]
    out["reactions_with_changed_flux"] = int(len(changed))
    if len(changed) > top_n and session_dir:
        out_dir = os.path.join(session_dir, label)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"{label}_flux_changes.csv")
        table.round(6).to_csv(csv_path, index=False)
        out["full_flux_table"] = csv_path
    return out


def run_moma(knockout_reactions=None, knockout_genes=None, linear: bool = True) -> dict:
    """
    Assess the metabolic impact of a knockout using MOMA
    (Minimization Of Metabolic Adjustment).

    Core cobrapy function: cobra.flux_analysis.moma.moma.
    Computes the wild-type flux distribution first, then knocks out the given
    genes/reactions and finds the feasible flux distribution closest to wild type.
    At least one of knockout_reactions / knockout_genes must be provided.
    """
    from cobra.flux_analysis import moma as _moma
    try:
        model = model_manager.get_current_model()
        if isinstance(model, dict):
            return {"error": "No model is currently loaded. Please load a model first."}
    except Exception as e:
        return {"error": str(e)}

    if not model.objective.expression:
        return {"error": "No objective function is set on the model. Set an objective before running MOMA."}

    _apply_uploaded_bounds(model)

    gene_objs, rxn_objs, errors = _resolve_knockouts(model, knockout_genes, knockout_reactions)
    if not gene_objs and not rxn_objs:
        return {"error": "Provide at least one valid knockout target via knockout_genes or knockout_reactions.", "unresolved": errors}

    try:
        wt = model.optimize()
        if wt.status != "optimal":
            return {"error": f"Wild-type optimisation was not optimal (status: {wt.status})."}

        obj_coeffs = _objective_reaction_coeffs(model)
        wt_growth = sum(c * wt.fluxes[r.id] for r, c in obj_coeffs.items())

        ko_model = model.copy()
        for g in gene_objs:
            ko_model.genes.get_by_id(g.id).knock_out()
        for r in rxn_objs:
            ko_model.reactions.get_by_id(r.id).knock_out()
        sol = _moma(ko_model, solution=wt, linear=linear)

        if sol.status != "optimal":
            return {"error": f"MOMA did not reach an optimal solution (status: {sol.status})."}

        ko_growth = sum(c * sol.fluxes[r.id] for r, c in obj_coeffs.items())

        result = {
            "status": sol.status,
            "method": "linear MOMA" if linear else "quadratic MOMA",
            "knocked_out_genes": [g.id for g in gene_objs],
            "knocked_out_reactions": [r.id for r in rxn_objs],
            "wild_type_growth": _clip_zero(round(float(wt_growth), 6)),
            "knockout_growth": _clip_zero(round(float(ko_growth), 6)),
            "growth_ratio": round(float(_clip_zero(ko_growth) / wt_growth), 4) if wt_growth else None,
            "moma_distance": round(float(sol.objective_value), 6),
        }
        result.update(_flux_change_report(wt, sol, "moma"))
        if errors:
            result["unresolved_targets"] = errors
        return result
    except Exception as e:
        return {"error": str(e)}


def run_room(
    knockout_reactions=None,
    knockout_genes=None,
    linear: bool = True,
    delta: float = 0.03,
    epsilon: float = 0.001,
) -> dict:
    """
    Assess the metabolic impact of a knockout using ROOM
    (Regulatory On/Off Minimization).

    Core cobrapy function: cobra.flux_analysis.room.room.
    Computes the wild-type flux distribution first, then knocks out the given
    genes/reactions and finds the flux distribution that changes the on/off
    state of the fewest reactions relative to wild type.
    At least one of knockout_reactions / knockout_genes must be provided.
    """
    from cobra.flux_analysis import room as _room
    try:
        model = model_manager.get_current_model()
        if isinstance(model, dict):
            return {"error": "No model is currently loaded. Please load a model first."}
    except Exception as e:
        return {"error": str(e)}

    if not model.objective.expression:
        return {"error": "No objective function is set on the model. Set an objective before running ROOM."}

    _apply_uploaded_bounds(model)

    gene_objs, rxn_objs, errors = _resolve_knockouts(model, knockout_genes, knockout_reactions)
    if not gene_objs and not rxn_objs:
        return {"error": "Provide at least one valid knockout target via knockout_genes or knockout_reactions.", "unresolved": errors}

    try:
        wt = model.optimize()
        if wt.status != "optimal":
            return {"error": f"Wild-type optimisation was not optimal (status: {wt.status})."}

        obj_coeffs = _objective_reaction_coeffs(model)
        wt_growth = sum(c * wt.fluxes[r.id] for r, c in obj_coeffs.items())

        ko_model = model.copy()
        for g in gene_objs:
            ko_model.genes.get_by_id(g.id).knock_out()
        for r in rxn_objs:
            ko_model.reactions.get_by_id(r.id).knock_out()
        sol = _room(ko_model, solution=wt, linear=linear, delta=delta, epsilon=epsilon)

        if sol.status != "optimal":
            return {"error": f"ROOM did not reach an optimal solution (status: {sol.status})."}

        ko_growth = sum(c * sol.fluxes[r.id] for r, c in obj_coeffs.items())

        result = {
            "status": sol.status,
            "method": "linear ROOM" if linear else "ROOM (MILP)",
            "delta": delta,
            "epsilon": epsilon,
            "knocked_out_genes": [g.id for g in gene_objs],
            "knocked_out_reactions": [r.id for r in rxn_objs],
            "wild_type_growth": _clip_zero(round(float(wt_growth), 6)),
            "knockout_growth": _clip_zero(round(float(ko_growth), 6)),
            "growth_ratio": round(float(_clip_zero(ko_growth) / wt_growth), 4) if wt_growth else None,
        }
        result.update(_flux_change_report(wt, sol, "room"))
        if errors:
            result["unresolved_targets"] = errors
        return result
    except Exception as e:
        return {"error": str(e)}


def find_blocked_rxns(
    reaction_names=None,
    open_exchanges: bool = False,
    zero_cutoff: float = None,
) -> dict:
    """
    Find reactions in the loaded model that cannot carry any flux.

    Core cobrapy function: cobra.flux_analysis.find_blocked_reactions.
    Read-only — the model is not modified. Whether a reaction is blocked depends
    on the exchange settings; set open_exchanges=True to open all exchanges first.
    """
    from cobra.flux_analysis import find_blocked_reactions as _fbr
    try:
        model = model_manager.get_current_model()
        if isinstance(model, dict):
            return {"error": "No model is currently loaded. Please load a model first."}
    except Exception as e:
        return {"error": str(e)}

    _apply_uploaded_bounds(model)

    # Resolve an optional reaction subset (ids or names, fuzzy matched).
    reaction_list = None
    unresolved = []
    if reaction_names:
        if isinstance(reaction_names, str):
            reaction_names = [r.strip() for r in reaction_names.split(",") if r.strip()]
        reaction_list = []
        seen = set()
        for name in reaction_names:
            tag, payload = _find_reaction(model, name)
            r = payload if tag == "exact" else (payload[0] if payload else None)
            if r is None:
                unresolved.append(name)
            elif r.id not in seen:
                reaction_list.append(r)
                seen.add(r.id)
        if not reaction_list:
            return {"error": "None of the provided reactions are valid in this model.", "unresolved": unresolved}

    try:
        blocked = _fbr(
            model,
            reaction_list=reaction_list,
            zero_cutoff=zero_cutoff,
            open_exchanges=open_exchanges,
        )
    except Exception as e:
        return {"error": str(e)}

    blocked_ids = sorted(r.id if hasattr(r, "id") else str(r) for r in blocked)
    considered = len(reaction_list) if reaction_list is not None else len(model.reactions)

    base = {
        "blocked_reaction_count": len(blocked_ids),
        "reactions_considered": considered,
        "open_exchanges": open_exchanges,
    }
    if unresolved:
        base["unresolved_reactions"] = unresolved

    if len(blocked_ids) > 5:
        import pandas as _pd
        out_dir = os.path.join(session_dir or ".", "blocked")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "blocked_reactions.csv")
        _pd.DataFrame({"reaction_id": blocked_ids}).to_csv(csv_path, index=False)
        base["first_5"] = blocked_ids[:5]
        base["message"] = f"{len(blocked_ids)} blocked reactions found. Full list saved to {csv_path}."
        return base
    base["blocked_reactions"] = blocked_ids
    return base


def remove_genes(gene_names, remove_reactions: bool = True) -> dict:
    """
    Permanently remove genes from the loaded model and simplify gene-reaction rules.

    Core cobrapy function: cobra.manipulation.delete.remove_genes.
    Unlike a knockout (which only constrains flux), this deletes the genes from
    the model object itself. The modified model REPLACES the loaded one. With
    remove_reactions=True, reactions left with no gene support are removed too.
    """
    from cobra.manipulation import remove_genes as _remove_genes
    try:
        model = model_manager.get_current_model()
        if isinstance(model, dict):
            return {"error": "No model is currently loaded. Please load a model first."}
    except Exception as e:
        return {"error": str(e)}

    if isinstance(gene_names, str):
        gene_names = [g.strip() for g in gene_names.split(",") if g.strip()]

    _FUZZY_THRESHOLD = 0.8
    valid_genes, not_found, fuzzy_notes, seen = [], [], [], set()
    for name in gene_names:
        tag, payload = _find_gene(model, name)
        if tag == "exact":
            gene = payload
        else:
            best = payload[0] if payload else None
            if best is None:
                not_found.append({"gene": name, "suggestions": []})
                continue
            q = name.strip().lower()
            score = max(_fuzzy_score(q, best.id.lower()), _fuzzy_score(q, best.name.lower()))
            if score < _FUZZY_THRESHOLD:
                not_found.append({"gene": name, "suggestions": [g.id for g in payload]})
                continue
            gene = best
            fuzzy_notes.append(f"'{name}' was fuzzy-matched to '{best.id}' (similarity {score:.0%})")
        if gene.id not in seen:
            valid_genes.append(gene)
            seen.add(gene.id)

    if not_found:
        return {
            "error": f"Gene(s) not found (no match above 80% similarity): {[e['gene'] for e in not_found]}",
            "suggestions": not_found,
        }
    if not valid_genes:
        return {"error": "None of the provided genes are valid in this model."}

    genes_before = len(model.genes)
    reactions_before = len(model.reactions)
    removed_gene_ids = [g.id for g in valid_genes]

    try:
        _remove_genes(model, valid_genes, remove_reactions=remove_reactions)
    except Exception as e:
        return {"error": str(e)}

    # remove_genes mutates the model in place; it is the same object stored in
    # model_manager.models, so the session already reflects the change.
    result = {
        "status": "success",
        "model_id": model_manager.current_model_id,
        "genes_requested_for_removal": removed_gene_ids,
        "genes_before": genes_before,
        "genes_after": len(model.genes),
        "genes_removed": genes_before - len(model.genes),
        "reactions_before": reactions_before,
        "reactions_after": len(model.reactions),
        "reactions_removed": reactions_before - len(model.reactions),
        "remove_reactions": remove_reactions,
        "note": "Genes were permanently removed from the loaded model and GPRs were simplified.",
    }
    if fuzzy_notes:
        result["fuzzy_matches"] = fuzzy_notes
    return result


def gapfill(
    universal_model_file: str = None,
    lower_bound: float = 0.05,
    demand_reactions: bool = True,
    exchange_reactions: bool = False,
    iterations: int = 1,
) -> dict:
    """
    Find the minimal set of reactions that restores growth to the loaded model
    (analysis-oriented gap filling).

    Core cobrapy function: cobra.flux_analysis.gapfilling.gapfill.
    Candidate reactions are drawn from a 'universal' model: either an uploaded
    SBML file (universal_model_file) or, if none is given, the BiGG universal
    model (downloaded on demand — requires internet). The loaded model is NOT
    modified; the suggested reactions are reported.
    """
    from cobra.flux_analysis import gapfill as _gapfill
    from cobra.io import read_sbml_model as _read_sbml, load_json_model as _load_json
    try:
        model = model_manager.get_current_model()
        if isinstance(model, dict):
            return {"error": "No model is currently loaded. Please load a model first."}
    except Exception as e:
        return {"error": str(e)}

    _apply_uploaded_bounds(model)

    # ── Obtain the universal model ──────────────────────────────────────────────
    universal = None
    universal_source = None
    if universal_model_file:
        path = universal_model_file
        if not os.path.isabs(path) and session_uploads_dir:
            path = os.path.join(session_uploads_dir, path)
        if not os.path.exists(path):
            return {"error": f"Universal model file not found: {path}"}
        try:
            universal = _read_sbml(path)
            universal_source = f"uploaded file ({os.path.basename(path)})"
        except Exception as e:
            return {"error": f"Could not read universal model SBML: {e}"}
    else:
        # Fall back to the BiGG universal model (downloaded and cached locally).
        import requests as _requests
        cache_dir = os.path.join(session_dir, "gapfill") if session_dir else "gapfill"
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "bigg_universal_model.json")
        if not os.path.exists(cache_path):
            url = "http://bigg.ucsd.edu/static/namespace/universal_model.json"
            try:
                resp = _requests.get(url, timeout=60)
                resp.raise_for_status()
                with open(cache_path, "wb") as fh:
                    fh.write(resp.content)
            except Exception as e:
                return {"error": (
                    "No universal model file was provided and the BiGG universal model "
                    f"could not be downloaded ({e}). Upload a universal SBML file instead."
                )}
        try:
            universal = _load_json(cache_path)
            universal_source = "BiGG universal model"
        except Exception as e:
            return {"error": f"Could not load the BiGG universal model: {e}"}

    # ── Run gap filling ─────────────────────────────────────────────────────────
    try:
        solutions = _gapfill(
            model,
            universal=universal,
            lower_bound=lower_bound,
            demand_reactions=demand_reactions,
            exchange_reactions=exchange_reactions,
            iterations=iterations,
        )
    except Exception as e:
        return {"error": (
            f"Gap filling failed: {e}. This often means the identifiers in the universal "
            "model do not match the loaded model's namespace, or no solution exists for the "
            "requested lower_bound."
        )}

    runs = []
    for i, rxn_set in enumerate(solutions, start=1):
        runs.append({
            "iteration": i,
            "reaction_count": len(rxn_set),
            "reactions": [
                {"id": r.id, "name": r.name, "reaction": r.build_reaction_string()}
                for r in rxn_set
            ],
        })

    return {
        "status": "success",
        "model_id": model_manager.current_model_id,
        "universal_source": universal_source,
        "lower_bound": lower_bound,
        "demand_reactions": demand_reactions,
        "exchange_reactions": exchange_reactions,
        "iterations": iterations,
        "solutions": runs,
        "note": (
            "These reactions would restore growth if added to the model. The loaded model "
            "was not modified; add the chosen reactions with add_reaction if desired."
        ),
    }


def gapfill_model_with_carveme(
    media: str,
    mediadb: str = None,
    universe: str = None,
    model_id: str = None,
) -> dict:
    """
    Gap-fill the currently loaded model for one or more growth media using
    CarveMe's `gapfill` command (reconstruction-oriented gap filling).

    CarveMe CLI: gapfill <input.xml> -m <media> --fbc2 [--mediadb <file>]
                 [-u {grampos,gramneg}] -o <output.xml>
    (media is required; universe defaults to 'bacteria'.)

    The loaded model is written to SBML (fbc2 format, hence --fbc2), gap-filled
    against CarveMe's BiGG-based universe so it can grow on the given media, then
    loaded back into the session. Requires CarveMe to be installed and works best
    on CarveMe/BiGG-namespace models.
    """
    import subprocess as _sp

    try:
        model = model_manager.get_current_model()
        if isinstance(model, dict):
            return {"error": "No model is currently loaded. Please load a model first."}
    except Exception as e:
        return {"error": str(e)}

    if not media or not str(media).strip():
        return {"error": "media is required, e.g. 'M9' or 'LB,M9' (comma-separated growth media)."}
    # _coerce in executor.py turns a comma-separated string into a list; CarveMe
    # expects a single comma-separated string on the command line.
    if isinstance(media, list):
        media = ",".join(str(m).strip() for m in media)

    output_dir = os.path.join(session_dir, "carveme_gapfill") if session_dir else "carveme_gapfill"
    os.makedirs(output_dir, exist_ok=True)

    if not model_id:
        model_id = f"{model_manager.current_model_id}_gapfilled"

    input_path = os.path.join(output_dir, f"{model_manager.current_model_id}_input.xml")
    out_path = os.path.join(output_dir, f"{model_id}.xml")
    try:
        write_sbml_model(model, input_path)
    except Exception as e:
        return {"error": f"Could not export the loaded model to SBML for gap filling: {e}"}

    # write_sbml_model produces SBML in fbc2 format, so tell CarveMe's gapfill
    # to read it as fbc2 (its --fbc2/--cobra flags declare the input format).
    cmd = ["gapfill", input_path, "-m", media, "-o", out_path, "--fbc2"]
    if mediadb:
        mediadb_path = mediadb
        if not os.path.isabs(mediadb_path) and session_uploads_dir:
            mediadb_path = os.path.join(session_uploads_dir, mediadb_path)
        if not os.path.exists(mediadb_path):
            return {"error": f"Media database file not found: {mediadb_path}"}
        cmd += ["--mediadb", mediadb_path]
    if universe:
        cmd += ["-u", universe]

    try:
        result = _sp.run(cmd, capture_output=True)
    except FileNotFoundError:
        return {"error": "CarveMe (gapfill) not found. Install with: pip install carveme"}

    if result.returncode != 0 or not os.path.exists(out_path):
        stderr = result.stderr.decode(errors="replace").strip() if result.stderr else ""
        stdout = result.stdout.decode(errors="replace").strip() if result.stdout else ""
        return {"error": "CarveMe gap filling failed to produce an output model.",
                "detail": stderr or stdout or "No output captured."}

    rxns_before = len(model.reactions)
    try:
        loaded_id = model_manager.load_sbml(out_path)
        new_model = model_manager.get_current_model()
        return {
            "status": "success",
            "model_id": loaded_id,
            "media": media,
            "universe": universe or "default",
            "output_file": out_path,
            "reactions_before": rxns_before,
            "reactions_after": len(new_model.reactions),
            "reactions_added": len(new_model.reactions) - rxns_before,
            "metabolites": len(new_model.metabolites),
            "genes": len(new_model.genes),
            "note": "The gap-filled model has been loaded into the session.",
        }
    except Exception as e:
        return {
            "status": "success",
            "warning": f"Model gap-filled but could not be auto-loaded: {e}",
            "media": media,
            "output_file": out_path,
        }


# ── Planner metadata ──────────────────────────────────────────────────────────
# Each tool function used by the planner carries a _planner_meta attribute.
# planner/tool_registry.py reads these to build TOOL_FN_MAP, TOOL_USER_INPUTS,
# and AVAILABLE_TOOLS_TEXT — the single source of truth for all three.

load_model._planner_meta = {
    "name": "load_model",
    "description": "Load a metabolic model into the session by organism name or model ID (auto-resolved from BiGG and BioModels) or from a locally uploaded SBML file — must be done before any analysis step.",
    "params": {
        "model_id": {
            "description": "Pass exactly what the user typed — a model ID or organism name, verbatim. Do not translate or infer.",
            "default": None,
            "required": True,
        }
    },
}

run_fba._planner_meta = {
    "name": "run_fba",
    "description": "Optimise the model's objective (typically biomass/growth) and compute steady-state fluxes for every reaction — the foundational step for most metabolic analyses.",
    "params": {},
}

run_fva._planner_meta = {
    "name": "run_fva",
    "description": "Compute the minimum and maximum achievable flux for every reaction while holding the objective at a set fraction of its optimum — reveals which reactions are fixed, flexible, or blocked under the current constraints.",
    "params": {
        "rxn_names": {
            "description": "Specific reaction names to analyze. Leave empty to run FVA on all reactions.",
            "default": None,
            "required": False,
        },
        "fraction_of_optimum": {
            "description": "Fraction of the optimal objective value to maintain (default: 0.9)",
            "default": 0.9,
            "required": False,
        },
    },
}

gene_knockout_simulation._planner_meta = {
    "name": "gene_knockout_simulation",
    "description": "Simulate the deletion of one or more genes by constraining their associated reactions to zero flux (following GPR rules) — internally runs FBA and directly returns the post-knockout growth rate alongside the wild-type growth rate. No separate FBA step is needed after calling this tool; the growth result is already included in the output.",
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
    "description": "Simulate the removal of one or more reactions from the network — internally runs FBA and directly returns the post-knockout growth rate alongside the wild-type growth rate. No separate FBA step is needed after calling this tool; the growth result is already included in the output.",
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
    "description": "Draw random steady-state flux distributions from the feasible space using ACHR or OptGP — captures metabolic flexibility and variability that a single FBA optimum cannot reveal.",
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
    "description": "Define which reaction to maximise or minimise before running FBA or pFBA — required when the model has no default objective or when switching between optimisation targets (e.g. ATP production vs. biomass).",
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
    "description": "Overlay FBA flux values onto an interactive Escher pathway map (saved as HTML) — makes it easy to see which reactions are active, saturated, or inactive, and to communicate results visually. Auto-detects the best map for the loaded model.",
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
    "description": "List all Escher pathway map names available for the loaded model — use when auto-detection fails or to browse which maps exist before calling visualize_escher.",
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
    "description": "Run Geometric FBA (gFBA) to obtain a unique, centred flux solution by iteratively tightening the feasible flux polytope — avoids the arbitrary optima of standard FBA and gives a single representative state for the network.",
    "params": {},
}

run_memote_report._planner_meta = {
    "name": "run_memote_report",
    "description": "Score the model against the memote quality standard — checks annotation completeness, stoichiometric consistency, metabolic coverage, and SBO term usage, returning a pass/fail summary and a full HTML report.",
    "params": {},
}

add_reaction._planner_meta = {
    "name": "add_reaction",
    "description": "Extend the model by adding a new reaction with stoichiometry, flux bounds, gene-reaction rule, and subsystem — used for manual curation, adding missing pathways, or incorporating hypothetical reactions before analysis.",
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
    "description": "Constrain reactions to specific flux ranges to model nutrient availability, gene expression levels, or experimental conditions — accepts a single reaction or a CSV batch file.",
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
            "input_type": "file",
            "file_types": ["csv"],
        },
    },
}


build_model_with_carveme._planner_meta = {
    "name": "build_model_with_carveme",
    "description": "Reconstruct a draft genome-scale metabolic model from a protein or DNA FASTA file using CarveMe. Prodigal is run automatically if a DNA/genome sequence is detected.",
    "params": {
        "fasta_file": {
            "description": "Path to the input FASTA file — either protein sequences (.faa) or a genome sequence (.fna/.fa/.fasta). DNA input triggers automatic Prodigal gene calling.",
            "default": None,
            "required": True,
            "input_type": "file",
            "file_types": ["faa", "fasta", "fa", "fna"],
        },
        "model_id": {
            "description": "ID/name for the output model (used as filename stem). Defaults to the input filename stem if not provided.",
            "default": None,
            "required": False,
        },
        "gram": {
            "description": "Gram stain for CarveMe template selection: 'grampos' (Gram-positive) or 'gramneg' (Gram-negative). Leave empty to use the default universal template.",
            "default": None,
            "required": False,
        },
    },
}


build_context_model_with_corda._planner_meta = {
    "name": "build_context_model_with_corda",
    "description": (
        "Build a context-specific (e.g. tissue- or condition-specific) genome-scale "
        "model from the currently loaded base/reference model using CORDA. The user "
        "only needs to upload ONE CSV: raw gene expression, gene confidence scores, "
        "or reaction confidence scores — the type is auto-detected. Expression is "
        "binned into confidence classes automatically."
    ),
    "params": {
        "data_csv": {
            "description": (
                "Path to the uploaded CSV (relative paths resolve against the session "
                "uploads directory). Two columns: an id column and a value column. Gene "
                "ids may use any namespace the model annotates — Ensembl, Entrez, UniProt, "
                "RefSeq or gene symbols (version suffixes are stripped) — so files from "
                "GTEx/Expression Atlas/GEO often work directly. The value is either a "
                "continuous expression value (TPM/FPKM/counts) or an integer confidence in "
                "{-1,0,1,2,3}."
            ),
            "default": None,
            "required": True,
            "input_type": "file",
            "file_types": ["csv", "tsv", "txt"],
        },
        "data_type": {
            "description": (
                "How to interpret the CSV: 'auto' (default, recommended), 'expression', "
                "'gene_confidence', or 'reaction_confidence'."
            ),
            "default": "auto",
            "required": False,
        },
        "model_id": {
            "description": "ID/name for the reconstructed model. Defaults to '<base_model>_corda'.",
            "default": None,
            "required": False,
        },
        "high_percentile": {
            "description": "Expression percentile at/above which a gene is high confidence (3). Default 75.",
            "default": 75.0,
            "required": False,
        },
        "mid_percentile": {
            "description": "Expression percentile for medium confidence (2). Default 50.",
            "default": 50.0,
            "required": False,
        },
        "low_percentile": {
            "description": "Expression percentile for low confidence (1); below it genes are treated as not expressed (-1). Default 25.",
            "default": 25.0,
            "required": False,
        },
        "keep_objective_high": {
            "description": "Force the model's objective reaction(s) to high confidence so the context model can still grow. Default True.",
            "default": True,
            "required": False,
        },
        "met_prod": {
            "description": "Optional comma-separated metabolite IDs that the reconstructed model must be able to produce.",
            "default": None,
            "required": False,
        },
    },
}


find_essential_genes._planner_meta = {
    "name": "find_essential_genes",
    "description": "Systematically knock out every gene in the model and report those whose deletion abolishes growth — identifies potential drug targets and genes indispensable for viability.",
    "params": {
        "threshold": {
            "description": "Minimal objective (growth) flux to be considered viable. Leave empty to use the default (1% of the maximal objective).",
            "default": None,
            "required": False,
        }
    },
}

find_essential_reactions._planner_meta = {
    "name": "find_essential_reactions",
    "description": "Systematically knock out every reaction and report those whose removal abolishes growth — reveals metabolic bottlenecks and candidate targets for growth inhibition.",
    "params": {
        "threshold": {
            "description": "Minimal objective (growth) flux to be considered viable. Leave empty to use the default (1% of the maximal objective).",
            "default": None,
            "required": False,
        }
    },
}

check_model_consistency._planner_meta = {
    "name": "check_model_consistency",
    "description": "Run FASTCC to identify and remove flux-inconsistent (permanently blocked) reactions — produces a consistent subnetwork where every reaction can carry non-zero flux, a prerequisite for reliable FVA and flux sampling.",
    "params": {
        "flux_threshold": {
            "description": "Flux threshold used by FASTCC to decide consistency (default 1.0).",
            "default": 1.0,
            "required": False,
        }
    },
}

check_mass_balance._planner_meta = {
    "name": "check_mass_balance",
    "description": "Verify that every reaction conserves mass and charge — unbalanced reactions can inflate flux values or mask infeasibility and should be corrected before analysis.",
    "params": {},
}

prune_unused_reactions._planner_meta = {
    "name": "prune_unused_reactions",
    "description": "Remove dead-end reactions (those with no metabolites assigned) that can arise during reconstruction — cleans up the model in-place before analysis.",
    "params": {},
}

prune_unused_metabolites._planner_meta = {
    "name": "prune_unused_metabolites",
    "description": "Remove orphaned metabolites not participating in any reaction — tidies the model in-place after manual edits or gene/reaction deletions.",
    "params": {},
}

find_minimal_medium._planner_meta = {
    "name": "find_minimal_medium",
    "description": "Identify the smallest set of nutrient imports that sustains a target growth rate — useful for designing minimal growth media, predicting auxotrophies, and reducing medium complexity.",
    "params": {
        "min_objective_value": {
            "description": "Minimum growth rate (objective value) the medium must support (default 0.1).",
            "default": 0.1,
            "required": False,
        },
        "minimize_components": {
            "description": "If True, minimise the NUMBER of medium components instead of total import flux (slower, mixed-integer). Default False.",
            "default": False,
            "required": False,
        },
        "open_exchanges": {
            "description": "If True, ignore current bounds and open all exchange reactions before searching. Default False.",
            "default": False,
            "required": False,
        },
    },
}

build_model_with_mackinac._planner_meta = {
    "name": "build_model_with_mackinac",
    "description": "Reconstruct a genome-scale model from a PATRIC genome via ModelSEED using Mackinac (requires PATRIC credentials and internet), then load it as a COBRA model",
    "params": {
        "genome_id": {
            "description": "PATRIC genome ID to reconstruct a model for (e.g. '226186.12').",
            "default": None,
            "required": True,
        },
        "username": {
            "description": "PATRIC/ModelSEED account username for authentication.",
            "default": None,
            "required": True,
        },
        "password": {
            "description": "PATRIC/ModelSEED account password for authentication.",
            "default": None,
            "required": True,
        },
        "source": {
            "description": "Genome source database: 'patric' (default) or 'rast'.",
            "default": "patric",
            "required": False,
        },
        "model_id": {
            "description": "ID/name for the reconstructed model. Defaults to '<genome_id>_modelseed'.",
            "default": None,
            "required": False,
        },
    },
}

run_moma._planner_meta = {
    "name": "run_moma",
    "description": "Predict the post-knockout flux state by finding the distribution closest to wild type (minimising flux re-routing) — more biologically realistic than FBA re-optimisation when the cell cannot fully adapt to the perturbation.",
    "params": {
        "knockout_reactions": {
            "description": "Reaction IDs/names to knock out (single value or comma-separated). At least one knockout target (reactions or genes) is required.",
            "default": None,
            "required": False,
        },
        "knockout_genes": {
            "description": "Gene IDs/names to knock out (single value or comma-separated). At least one knockout target (reactions or genes) is required.",
            "default": None,
            "required": False,
        },
        "linear": {
            "description": "Use linear (L1, faster) MOMA instead of quadratic (L2). Default True.",
            "default": True,
            "required": False,
        },
    },
}

run_room._planner_meta = {
    "name": "run_room",
    "description": "Predict the post-knockout flux state by minimising the number of reactions that switch on/off relative to wild type — models the assumption that cells minimise regulatory changes, complementing MOMA's flux-distance approach.",
    "params": {
        "knockout_reactions": {
            "description": "Reaction IDs/names to knock out (single value or comma-separated). At least one knockout target (reactions or genes) is required.",
            "default": None,
            "required": False,
        },
        "knockout_genes": {
            "description": "Gene IDs/names to knock out (single value or comma-separated). At least one knockout target (reactions or genes) is required.",
            "default": None,
            "required": False,
        },
        "linear": {
            "description": "Use the linear ROOM relaxation instead of the exact MILP. Default True (faster); set to false to run exact MILP.",
            "default": True,
            "required": False,
        },
        "delta": {
            "description": "Relative tolerance range (additive). Default 0.03.",
            "default": 0.03,
            "required": False,
        },
        "epsilon": {
            "description": "Absolute tolerance range (multiplicative). Default 0.001.",
            "default": 0.001,
            "required": False,
        },
    },
}

find_blocked_rxns._planner_meta = {
    "name": "find_blocked_rxns",
    "description": "Identify reactions forced to carry zero flux under all conditions within current bounds — flags dead-end pathways, missing transporters, and connectivity gaps in the network. Read-only; does not modify the model.",
    "params": {
        "reaction_names": {
            "description": "Specific reaction IDs/names to check (single value or comma-separated). Leave empty to check all reactions.",
            "default": None,
            "required": False,
        },
        "open_exchanges": {
            "description": "Open all exchange reactions to high flux before testing (default False). Use when blockage is due to a closed medium.",
            "default": False,
            "required": False,
        },
        "zero_cutoff": {
            "description": "Flux value treated as effectively zero. Leave empty to use the model's tolerance.",
            "default": None,
            "required": False,
        },
    },
}

remove_genes._planner_meta = {
    "name": "remove_genes",
    "description": "Permanently delete genes from the model and rewrite all GPR rules to reflect the removal — use for reconstruction curation or correcting incorrect gene associations. Optionally also removes reactions that lose all gene support.",
    "params": {
        "gene_names": {
            "description": "Gene IDs/names to remove permanently (single value or comma-separated).",
            "default": None,
            "required": True,
        },
        "remove_reactions": {
            "description": "Also remove reactions left without any gene support after removal. Default True.",
            "default": True,
            "required": False,
        },
    },
}

gapfill._planner_meta = {
    "name": "gapfill",
    "description": "Identify the smallest set of reactions from a universal database that, when added, restores growth — pinpoints metabolic gaps from incomplete reconstruction. The loaded model is NOT modified; suggested reactions must be reviewed and added manually.",
    "params": {
        "universal_model_file": {
            "description": "Optional SBML file of candidate (universal) reactions. If omitted, the BiGG universal model is downloaded and used (requires internet).",
            "default": None,
            "required": False,
            "input_type": "file",
            "file_types": ["xml", "sbml"],
        },
        "lower_bound": {
            "description": "Minimum objective (growth) flux the gap-filled model must achieve. Default 0.05.",
            "default": 0.05,
            "required": False,
        },
        "demand_reactions": {
            "description": "Consider adding demand reactions for metabolites. Default True.",
            "default": True,
            "required": False,
        },
        "exchange_reactions": {
            "description": "Consider adding exchange (uptake) reactions for all metabolites. Default False.",
            "default": False,
            "required": False,
        },
        "iterations": {
            "description": "Number of gap-filling rounds; each finds an alternative solution. Default 1.",
            "default": 1,
            "required": False,
        },
    },
}

gapfill_model_with_carveme._planner_meta = {
    "name": "gapfill_model_with_carveme",
    "description": "Gap-fill the loaded model for given growth media using CarveMe's gapfill command (reconstruction-oriented). Adds reactions so the model grows on the media, then reloads it. Requires CarveMe; best on CarveMe/BiGG-namespace models",
    "params": {
        "media": {
            "description": "Growth media the model must grow on, e.g. 'M9' or 'LB,M9' (comma-separated). Required.",
            "default": None,
            "required": True,
        },
        "mediadb": {
            "description": "Optional media database file (CSV/TSV) defining custom media compositions.",
            "default": None,
            "required": False,
            "input_type": "file",
            "file_types": ["csv", "tsv"],
        },
        "universe": {
            "description": "CarveMe pre-built universe to draw candidate reactions from: 'grampos' or 'gramneg'. Leave empty for the default ('bacteria').",
            "default": None,
            "required": False,
        },
        "model_id": {
            "description": "ID/name for the gap-filled output model. Defaults to '<loaded_model>_gapfilled'.",
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
        "Loads a metabolic model from BiGG or BioModels. "
        "Pass EXACTLY what the user typed — do not translate, infer, or guess a model ID. "
        "If the user said 'rat', pass 'rat'. If they said 'e_coli_core', pass 'e_coli_core'. "
        "The tool handles ID resolution and search internally."
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
    description=(
        "Returns a sample list of reactions/genes/metabolites in the model. "
        "query must be 'reactions', 'genes', or 'metabolites'. "
        "Always returns at most 30 items — do NOT use this to find a specific reaction, gene, or "
        "metabolite; use reaction_info / gene_info / metabolite_info for that instead. "
        "Use this only to give the user a general sense of what is in the model."
    ),
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
run_fba_tool = FunctionTool.from_defaults(
    fn=run_fba,
    name="run_fba",
    description="Run Flux Balance Analysis on the currently loaded model.",
    return_direct=return_direct
)
set_objective_tool = FunctionTool.from_defaults(
    fn=set_model_objective,
    name="set_model_objective",
    description=(
        "Set the objective function for the currently loaded model. "
        "Provide a reaction ID (fuzzy matched if not exact) and an optimization direction "
        "('max' or 'min'). Must be called before running FBA."
    ),
    return_direct=return_direct
)
run_fva_tool = FunctionTool.from_defaults(
    fn=run_fva,
    name="run_fva",
    description=(
        "Run Flux Variability Analysis (FVA) on the currently loaded model. "
        "Optionally accepts a list of reaction names (rxn_names); if not provided or empty, "
        "runs FVA on all reactions. Also accepts fraction_of_optimum (default 0.9)."
    ),
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
    name="run_memote_report",
    description=(
        "Run the memote quality test suite on the currently loaded metabolic model. "
        "Returns a concise pass/fail summary with messages and saves a full HTML report "
        "to the session directory."
    ),
    return_direct=return_direct
)

run_pfba_tool = FunctionTool.from_defaults(
    fn=run_pfba,
    name="run_pfba",
    description=(
        "Run Parsimonious FBA (pFBA) on the currently loaded model. "
        "Maximises the objective while minimising total absolute flux to find the most "
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
    name="add_reaction",
    description=(
        "Add a new reaction to the currently loaded metabolic model. "
        "Accepts a reaction ID, equation string (e.g. '1.0 malACP_c + h_c --> co2_c + ACP_c'), "
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

build_model_with_carveme_tool = FunctionTool.from_defaults(
    fn=build_model_with_carveme,
    name="build_model_with_carveme",
    description=(
        "Reconstruct a draft genome-scale metabolic model from a FASTA file using CarveMe. "
        "Accepts protein sequences (.faa) or a whole genome (.fna/.fa) — DNA input is "
        "automatically handled by running Prodigal for gene calling (meta mode). "
        "The reconstructed model is saved as SBML and loaded into the active session."
    ),
    return_direct=return_direct,
)

build_context_model_with_corda_tool = FunctionTool.from_defaults(
    fn=build_context_model_with_corda,
    name="build_context_model_with_corda",
    description=(
        "Build a context-specific (tissue/condition-specific) genome-scale model from the "
        "currently loaded base/reference model using CORDA. The user uploads a single CSV — "
        "raw gene expression, gene confidence scores, or reaction confidence scores — and the "
        "type is auto-detected (expression is binned into CORDA confidence classes "
        "automatically). The reconstructed model is saved as SBML and loaded into the session."
    ),
    return_direct=return_direct,
)

find_essential_genes_tool = FunctionTool.from_defaults(
    fn=find_essential_genes,
    name="find_essential_genes",
    description=(
        "Finds the essential genes of the currently loaded model using "
        "cobra.flux_analysis.find_essential_genes. A gene is essential if knocking "
        "out all reactions that depend on it abolishes growth. Optionally accepts a "
        "growth threshold (defaults to 1% of the maximal objective). Requires an objective."
    ),
    return_direct=return_direct,
)

find_essential_reactions_tool = FunctionTool.from_defaults(
    fn=find_essential_reactions,
    name="find_essential_reactions",
    description=(
        "Finds the essential reactions of the currently loaded model using "
        "cobra.flux_analysis.find_essential_reactions. A reaction is essential if "
        "constraining its flux to zero abolishes growth. Optionally accepts a growth "
        "threshold (defaults to 1% of the maximal objective). Requires an objective."
    ),
    return_direct=return_direct,
)

check_model_consistency_tool = FunctionTool.from_defaults(
    fn=check_model_consistency,
    name="check_model_consistency",
    description=(
        "Checks the consistency of the loaded model with FASTCC "
        "(cobra.flux_analysis.fastcc) and removes blocked (flux-inconsistent) "
        "reactions. The cleaned consistent model replaces the loaded model. "
        "Optionally accepts a flux_threshold (default 1.0)."
    ),
    return_direct=return_direct,
)

check_mass_balance_tool = FunctionTool.from_defaults(
    fn=check_mass_balance,
    name="check_mass_balance",
    description=(
        "Checks the mass and charge balance of every reaction in the loaded model "
        "(cobra.manipulation.check_mass_balance) and reports the unbalanced reactions "
        "with their element/charge imbalance. An empty result means all reactions are balanced."
    ),
    return_direct=return_direct,
)

prune_unused_reactions_tool = FunctionTool.from_defaults(
    fn=prune_unused_reactions,
    name="prune_unused_reactions",
    description=(
        "Removes reactions with no assigned metabolites from the loaded model "
        "(cobra.manipulation.prune_unused_reactions). The pruned model replaces the loaded model."
    ),
    return_direct=return_direct,
)

prune_unused_metabolites_tool = FunctionTool.from_defaults(
    fn=prune_unused_metabolites,
    name="prune_unused_metabolites",
    description=(
        "Removes metabolites not involved in any reaction from the loaded model "
        "(cobra.manipulation.prune_unused_metabolites). The pruned model replaces the loaded model."
    ),
    return_direct=return_direct,
)

find_minimal_medium_tool = FunctionTool.from_defaults(
    fn=find_minimal_medium,
    name="find_minimal_medium",
    description=(
        "Finds the minimal growth medium for the loaded model "
        "(cobra.medium.minimal_medium) — the import fluxes/components needed to sustain "
        "at least a target growth rate. Accepts min_objective_value (default 0.1), "
        "minimize_components (minimise number of ingredients, slower), and open_exchanges."
    ),
    return_direct=return_direct,
)

build_model_with_mackinac_tool = FunctionTool.from_defaults(
    fn=build_model_with_mackinac,
    name="build_model_with_mackinac",
    description=(
        "Reconstructs a genome-scale metabolic model from a PATRIC genome via the "
        "ModelSEED web service using Mackinac, then loads it as a COBRA model. Requires "
        "a PATRIC genome_id, account username and password, and live internet access. "
        "The reconstructed model is saved as SBML and loaded into the active session."
    ),
    return_direct=return_direct,
)

request_file_upload_tool = FunctionTool.from_defaults(
    fn=request_file_upload,
    name="request_file_upload",
    description=(
        "Call this tool when you need the user to upload a file before you can proceed. "
        "Provide param (the exact parameter name the file maps to), "
        "file_types (comma-separated accepted extensions, e.g. 'csv,tsv' or 'faa,fasta,fa,fna'), "
        "and description (one sentence explaining what the file must contain). "
        "After calling this tool, stop immediately — do not call any other tool in the same turn."
    ),
    return_direct=return_direct,
)

request_tool_inputs_tool = FunctionTool.from_defaults(
    fn=request_tool_inputs,
    name="request_tool_inputs",
    description=(
        "Call this tool when you know which tool to run but the user has not provided all required "
        "arguments and they cannot be safely inferred. "
        "Provide tool_name (exact tool to call, e.g. 'add_reaction'), "
        "prefilled_params (JSON string of args derivable from the user's message, e.g. "
        "'{\"reaction_id\": \"GALK\"}', or '{}' if none), "
        "and explanation (one sentence on what you understood from the user's request). "
        "After calling this tool, stop immediately — do not call any other tool in the same turn."
    ),
    return_direct=return_direct,
)

run_moma_tool = FunctionTool.from_defaults(
    fn=run_moma,
    name="run_moma",
    description=(
        "Assesses the metabolic impact of a knockout using MOMA "
        "(cobra.flux_analysis.moma). Computes the wild-type flux distribution, "
        "knocks out the given genes and/or reactions, and finds the feasible flux "
        "distribution closest to wild type. Accepts knockout_reactions, knockout_genes "
        "(at least one required) and linear (default True)."
    ),
    return_direct=return_direct,
)

run_room_tool = FunctionTool.from_defaults(
    fn=run_room,
    name="run_room",
    description=(
        "Assesses the metabolic impact of a knockout using ROOM "
        "(cobra.flux_analysis.room). Minimises the number of reactions whose on/off "
        "state changes relative to wild type after knocking out the given genes and/or "
        "reactions. Accepts knockout_reactions, knockout_genes (at least one required), "
        "linear (default False), delta (0.03) and epsilon (0.001)."
    ),
    return_direct=return_direct,
)

find_blocked_rxns_tool = FunctionTool.from_defaults(
    fn=find_blocked_rxns,
    name="find_blocked_rxns",
    description=(
        "Finds reactions in the loaded model that cannot carry any flux "
        "(cobra.flux_analysis.find_blocked_reactions). Read-only. Optionally accepts a "
        "subset of reaction_names, open_exchanges (open all exchanges first, default False) "
        "and zero_cutoff (flux treated as zero)."
    ),
    return_direct=return_direct,
)

remove_genes_tool = FunctionTool.from_defaults(
    fn=remove_genes,
    name="remove_genes",
    description=(
        "Permanently removes genes from the loaded model and simplifies gene-reaction "
        "rules (cobra.manipulation.remove_genes). Unlike a knockout, the genes are deleted "
        "from the model itself; with remove_reactions=True (default) reactions left without "
        "gene support are also removed. The modified model replaces the loaded one."
    ),
    return_direct=return_direct,
)

gapfill_tool = FunctionTool.from_defaults(
    fn=gapfill,
    name="gapfill",
    description=(
        "Finds the minimal set of reactions that restores growth to the loaded model "
        "(cobra.flux_analysis.gapfill). Candidate reactions come from an uploaded universal "
        "SBML file, or the BiGG universal model if none is provided (downloaded on demand). "
        "Reports the suggested reactions; the loaded model is not modified."
    ),
    return_direct=return_direct,
)

gapfill_model_with_carveme_tool = FunctionTool.from_defaults(
    fn=gapfill_model_with_carveme,
    name="gapfill_model_with_carveme",
    description=(
        "Gap-fills the currently loaded model for one or more growth media using CarveMe's "
        "`gapfill` command (reconstruction-oriented). Adds reactions from CarveMe's BiGG-based "
        "universe so the model can grow on the given media, then reloads the gap-filled model "
        "into the session. Requires CarveMe and works best on CarveMe/BiGG-namespace models."
    ),
    return_direct=return_direct,
)






# TEST CODE

# model_manager = ModelManager()
# model_manager.load_model_by_id('e_coli_core')
# model_manager.bounds_dict = pd.read_csv("bounds_data/e_coli_bounds.csv").values.tolist()
# print(run_fba())