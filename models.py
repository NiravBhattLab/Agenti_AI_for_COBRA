import cobra
from cobra.io import read_sbml_model
from cobra.io.web.load import load_model, BiGGModels, BioModels

class ModelManager:
    def __init__(self):
        self.models = {}
        self.current_model_id = None
        self.bounds_data = None
        self.objective = False

    def load_model_by_id(self, model_id):
        if "xml" not in model_id:
            model_id = f"{model_id}.xml"
        base_model_id = model_id.split(".")[0]

        try:
            repositories = [BioModels(), BiGGModels()]
            model = load_model(base_model_id, repositories=repositories)
            if model:
                self.models[base_model_id] = model
                self.current_model_id = base_model_id
                if model.objective:
                    self.objective = True
                return base_model_id
        except Exception as e:
            return {"response": f"Error loading from remote repositories: {e}"}

    def load_sbml(self, file_path):
        model_id = str(file_path).split("/")[-1].split(".")[0]
        model = read_sbml_model(file_path)
        # model_oject = Model(model, model_id)
        self.models[model_id] = model
        self.current_model_id = model_id
        if model.objective:
            self.objective = True
        return model_id

    def get_current_model(self):
        if not self.current_model_id:
            return {"response": "No model is currently loaded."}
        return self.models[self.current_model_id]

    def set_current_model(self, model_id):
        if model_id not in self.models:
            return {"response": "Invalid model ID."}
        self.current_model_id = model_id   





# class Model:
#     def __init__(self, model, model_id):
#         self.model = model
#         self.model_id = model_id
#         self.bounds_data = None
#         self.objective = None

#     def set_bounds_dict(self, bounds_dict):
#         self.bounds_dict = bounds_dict


# if __name__ == "__main__":
    # Example usage
    # manager = ModelManager()
    # model_id = manager.load_sbml("uploads/BIOMD0000000173_url.xml")
    # print(f"Loaded model ID: {model_id}")
    
    # model = manager.get_current_model()
    # print("\n\n\n")
    # print(model)
    # data = {
    #         "model_id": str(model.id),
    #         # "model_name": str(model.name) if model.name else "unknown",
    #         "objective_reaction": str(model.objective.expression) if model.objective.expression else "Not Set Yet",
    #         "reactions_count": len(model.reactions),
    #         "metabolites_count": len(model.metabolites),
    #         "genes_count": len(model.genes),
    #     }
    # print(data)