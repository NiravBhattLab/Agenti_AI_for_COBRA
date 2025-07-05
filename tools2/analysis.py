from llama_index.core.tools import FunctionTool
from cobra.flux_analysis import flux_variability_analysis

return_direct = True

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