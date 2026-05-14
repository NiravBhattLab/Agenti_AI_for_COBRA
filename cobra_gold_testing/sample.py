from cobra.io import read_sbml_model

model = read_sbml_model("uploads/e_coli_core.xml")

print(f"Model ID: {model.id}")
print(f"Reactions: {len(model.reactions)}")
print(f"Metabolites: {len(model.metabolites)}")
print(f"Genes: {len(model.genes)}")
print()

# Important uptake reactions and the objective
reactions_of_interest = {
    "EX_glc__D_e": "Glucose uptake",
    "EX_o2_e":     "Oxygen uptake",
    "EX_nh4_e":    "Ammonium uptake",
    "EX_pi_e":     "Phosphate uptake",
    "EX_co2_e":    "CO2 exchange",
    "BIOMASS_Ecoli_core_w_GAM": "Biomass (objective)",
}

print("=== Reaction bounds ===")
for rxn_id, label in reactions_of_interest.items():
    try:
        rxn = model.reactions.get_by_id(rxn_id)
        print(f"  {label} ({rxn_id}):  lb={rxn.lower_bound:.1f}, ub={rxn.upper_bound:.1f}")
    except KeyError:
        print(f"  {label} ({rxn_id}): not found in model")
print()

# Run FBA
solution = model.optimize()

print("=== FBA Results ===")
print(f"  Status:           {solution.status}")
print(f"  Objective value:  {solution.objective_value:.6f}")
print()

if solution.status == "optimal":
    print("=== Key fluxes ===")
    for rxn_id, label in reactions_of_interest.items():
        if rxn_id in solution.fluxes:
            print(f"  {label} ({rxn_id}): {solution.fluxes[rxn_id]:.4f}")
