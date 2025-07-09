# import pandas as pd
# from cobra.io import read_sbml_model

# model = read_sbml_model(r"E:\INTERNSHIP\IITM\metabolic\uploads\e_coli_core.xml")

# # Function to run FBA
# def run_fba(bounds: dict = {}) -> str:
#     for rxn_id, (lval, uval) in bounds.items():
#         if rxn_id in [r.id for r in model.reactions]:
#             model.reactions.get_by_id(rxn_id).bounds = (lval, uval)
#     solution = model.optimize()
#     return f"Objective value: {solution.objective_value}"

# bounds = {
#     "EX_glc__D_e": (-10, 0),  # Glucose uptake
#     "EX_o2_e": (-20, 0),       # Oxygen uptake
# }

# print(model.objective.expression)
# result = run_fba(bounds)
# print(result)

from llama_index.llms.huggingface import HuggingFaceLLM
llm = HuggingFaceLLM(model_name="mistralai/Mistral-7B-Instruct-v0.3")