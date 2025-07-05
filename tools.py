from llama_index.core.tools import FunctionTool
from cobra.flux_analysis import flux_variability_analysis
from cobra.flux_analysis import single_gene_deletion, double_gene_deletion, single_reaction_deletion, double_reaction_deletion
from models import ModelManager
from ptypes import LoadModelInput
import pandas as pd
import os

return_direct = True
model_manager = None
def set_model_manager(manager):
    global model_manager
    model_manager = manager

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
        model_manager.load_model_by_id(model_id)
        return {"response": f"Model {model_id} loaded successfully.", "model_id": model_id}
    except Exception as e:
        return {"error" : str(e)}


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
            "Compartments": str([v for k,v in model.compartments.items()]),
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
            return {"reactions" : [rxn.name for rxn in model.reactions][:count]}
        elif query == "genes":
            return {"genes" : [gn.name for gn in model.genes][:count]}
        elif query == "metabolites":
            return {"metabolites" : [mb.name for mb in model.metabolites][:count]}
        else:
            return {
                "error": f"Unknown query: {query}",
            }
    except Exception as e:
        return {
            "error": str(e),
            "model_id": model_manager.current_model_id
        }


def reaction_info(rxn_name: str) -> dict:
    """
    Returns information about a specific reaction in the model.
    """
    try:
        model = model_manager.get_current_model()
        for rxn in model.reactions:
            if rxn.name == rxn_name:
                rxn_name = rxn.id
                break
        reaction = model.reactions.get_by_id(rxn_name)    
        return {
            "Reaction id": reaction.id,
            "name": reaction.name,
            "Stochiometry": reaction.build_reaction_string(),
            "GPR" : str(reaction.gpr) or "Not Set",
            "lower_bound": reaction.lower_bound,
            "upper_bound": reaction.upper_bound,
        }
    except Exception as e:
        return {"error": str(e)}


def metabolite_info(mb_id: str) -> dict:
    """
    Returns information about a specific metabolite in the model.
    """
    try:
        model = model_manager.get_current_model()
        metabolite = model.metabolites.get_by_id(mb_id)
        return {
            "Metabolite id": metabolite.id,
            "name": metabolite.name,
            "Formula": metabolite.formula,
            "Compartment": metabolite.compartment,
            "Total Reactions": len(metabolite.reactions),
            "Reactions": ', '.join([rxn.id for rxn in metabolite.reactions]),
        }
    except Exception as e:
        return {"error": str(e)}


def gene_info(gn_id: str) -> dict:
    """
    Returns information about a specific reaction in the model.
    This function simulates fetching reaction data from a database or API.
    """
    try:
        model = model_manager.get_current_model()
        gene = model.genes.get_by_id(gn_id)
        return {
            "Gene ID": gene.id,
            "name": gene.name,
            "Total Reactions": len(gene.reactions),
            "Reactions": ', '.join([rxn.id for rxn in gene.reactions]),
        }
    except Exception as e:
        return {"error": str(e)}

def run_fba() -> str:
    """
    Performs Flux Balance Analysis (FBA) on the current metabolic model.
    Returns Objective value and Model Status.
    """
    model = model_manager.get_current_model()
    bounds = model_manager.bounds_data

    if not bounds:
        return {"error": "No Bounds for model reactions are found."}
    if not model_manager.objective:
        return {"error": "No Objective Function is set for the model."}
    try:
        for rxn_id, lval, uval in bounds:
            if rxn_id in model.reactions:
                model.reactions.get_by_id(rxn_id).bounds = (lval, uval)
    except:
        return {"error": "Wrong Reaction bounds given."}
    
    solution = model.optimize()
    model_manager.objective = solution.objective_value
    return {
        "Objective value" : str(solution.objective_value),
        "status" : str(solution.status),
    }


def set_model_objective(objective_dict, direction="max"):
    """
    Sets the objective on the current model.
    """
    try:
        model = model_manager.get_current_model()

        cleaned_objective = {}
        for rxn_id, coeff in dict(objective_dict).items():
            if rxn_id not in model.reactions:
                return {"error": f"Reaction '{rxn_id}' not found in model."}
            rxn_obj = model.reactions.get_by_id(rxn_id)
            cleaned_objective[rxn_obj] = float(coeff)

        model.objective = cleaned_objective
        model.objective.direction = direction.lower()

        return {
            "status": "Objective set successfully.",
            "objective": model.objective.expression,
            "direction": model.objective.direction
        }

    except Exception as e:
        return {"error" : str(e)}

    
def run_fva(rxn_names, fraction_of_optimum=0.9):
    """
    Runs Flux Variability Analysis (FVA) on the model given a Reaction List and a Fraction of Optimum (FO) Value.    
    """
    try:
        model = model_manager.get_current_model()
        rxn_obj_list = []

        if not model_manager.objective:
            return {"error": "No Objective Function is set for the model."}

        for name in rxn_names:
            match = next(
                (rxn for rxn in model.reactions if name.lower() in rxn.name.lower()),
                None
            )
            if match is None:
                raise ValueError(f"Reaction name '{name}' not found in model.")
            rxn_obj_list.append(match)

        fva_result = flux_variability_analysis(model, rxn_obj_list, fraction_of_optimum=fraction_of_optimum)

        fva_df = fva_result.reset_index()
        fva_df.insert(0, "Reaction Name", [rxn.name for rxn in rxn_obj_list])
        fva_df.columns = ["Reaction Name", "Reaction ID", "Maximum Flux", "Minimum Flux"]

        if len(fva_df) > 5:
            output_dir = os.path.join(os.getcwd(), 'outputs')
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, f"fva_result.csv")
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


def gene_knockout_simulation(gene_names: list[str], type: str = "single") -> dict:
    """
    Performs single or double gene knockout simulations on the loaded metabolic model.
    """
    try:
        model = model_manager.get_current_model()
        valid_genes = []
        seen = set()
        for gene in model.reactions:
            if gene.name in gene_names and gene.id not in seen:
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
            file_path = "/tmp/gene_knockout_result.csv"
            result.to_csv(file_path, index=False)
            return {"file": file_path, "note": "Too many results to display. Download CSV."}

        return result.iloc[:5].to_dict(orient="records")

    except Exception as e:
        return {"error": str(e)}
    

def reaction_knockout_simulation(reaction_names: list[str], type: str = "single") -> dict:
    """
    Performs single or double reaction knockout simulations on the loaded metabolic model.
    """
    try:
        model = model_manager.get_current_model()
        valid_rxns = []
        seen = set()
        for rxn in model.reactions:
            if rxn.name in reaction_names and rxn.id not in seen:
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
            file_path = "/tmp/reaction_knockout_result.csv"
            result.to_csv(file_path, index=False)
            return {"file": file_path, "note": "Too many results to display. Download CSV."}

        return result.iloc[:5].to_dict(orient="records")

    except Exception as e:
        return {"error": str(e)}
    

######### TOOL SETUP

check_load_model_tool = FunctionTool.from_defaults(
    fn=check_model_loaded,
    name="check_model_loaded",
    description="Checks if a model is currently loaded in the ModelManager. Returns the model ID if loaded, otherwise an error message.",
    return_direct=return_direct
)

current_model_tool = FunctionTool.from_defaults(
    fn=get_current_model_id,
    name="get_cuurrent_model_id",
    description="""Returns the current model ID from the ModelManager.""",
    return_direct=return_direct
)

load_model_tool = FunctionTool.from_defaults(
    fn=load_model,
    name="load_model",
    description="Loads a model to be used for Analysis. It fetches a model from an API.",
    fn_schema=LoadModelInput,
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
    description="Returns the metadata related to a given Reaction by Name: reaction_id, name, Stochiometry, GPR, lower_bound, upper_bound",
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
    description="Runs Flux Variability Analysis (FVA) on the model given a Reaction List and a Fraction of Optimum (FO) Value",
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







# TEST CODE

# model_manager = ModelManager()
# model_manager.load_model_by_id('e_coli_core')
# file_path = r"E:\INTERNSHIP\IITM\metabolic\uploads\bounds_data\e_coli_bounds.csv"
# model_manager.bounds_dict = pd.read_csv(file_path).values.tolist()
# print(run_fba())

################## MISC

# def bounds_data_for_FBA(csv_file_path: str) -> dict:
#     """
#     Reads bounds data from a CSV file and returns it as a dictionary.
#     This function simulates fetching bounds data from a file.
#     """
#     try:
#         df = pd.read_csv(csv_file_path)
#         bounds_data = df.to_dict(orient='records')
#         model_manager.set_bounds_dict(bounds_data)
#         return bounds_data
#     except Exception as e:
#         return {"error": str(e)}

# set_bounds_tool = FunctionTool.from_defaults(
#     fn=bounds_data_for_FBA,
#     name="set_reaction_bounds_for_FBA",
#     description="Reads bounds data from a CSV file and returns it as a dictionary for Flux Balance Analysis (FBA)."
# )