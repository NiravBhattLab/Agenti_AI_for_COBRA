import cobra
import difflib
import requests
from cobra.io import read_sbml_model
from cobra.io.web.load import load_model, BiGGModels, BioModels
from cobra.io.sbml import validate_sbml_model

_BIGG_SEARCH_URL = "http://bigg.ucsd.edu/api/v2/search"
_BIOMODELS_SEARCH_URL = "https://biomodels.org/search"

# Common organism aliases → canonical scientific name fragment used in BiGG/BioModels.
_ORGANISM_ALIASES = {
    "mouse": "Mus musculus",
    "mice": "Mus musculus",
    "rat": "Rattus norvegicus",
    "human": "Homo sapiens",
    "yeast": "Saccharomyces cerevisiae",
    "e coli": "Escherichia coli",
    "ecoli": "Escherichia coli",
    "zebrafish": "Danio rerio",
    "fly": "Drosophila melanogaster",
    "worm": "Caenorhabditis elegans",
    "arabidopsis": "Arabidopsis thaliana",
    "tuberculosis": "Mycobacterium tuberculosis",
}


def _expand_query(query: str) -> str:
    """Replace common organism aliases with their scientific names."""
    return _ORGANISM_ALIASES.get(query.strip().lower(), query)


def _fuzzy_score(query: str, candidate: str) -> float:
    return difflib.SequenceMatcher(None, query.lower(), candidate.lower()).ratio()


def _bigg_match_score(query: str, bigg_id: str, organism: str) -> float:
    """Score query against bigg_id and organism independently; return the max.

    Scoring against the concatenated string causes false positives where
    fragments of the query match across the boundary between the two fields.
    """
    return max(_fuzzy_score(query, bigg_id), _fuzzy_score(query, organism))


def _search_bigg(query: str) -> str | None:
    """Return the best-matching BiGG model ID for query, or None.

    Uses the full model list instead of the search endpoint because BiGG's
    search API does not index by organism name and returns empty results for
    most natural-language queries.
    """
    query = _expand_query(query)
    try:
        resp = requests.get("http://bigg.ucsd.edu/api/v2/models", timeout=10)
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except Exception:
        return None
    if not results:
        return None
    scored = [
        (r["bigg_id"], _bigg_match_score(query, r.get("bigg_id", ""), r.get("organism", "")))
        for r in results
    ]
    best_id, best_score = max(scored, key=lambda x: x[1])
    # Require a minimum similarity to avoid false positives from Latin species name overlap.
    if best_score < 0.6:
        return None
    return best_id


def _search_biomodels(query: str) -> str | None:
    """Return the best-matching BioModels model ID for query, or None."""
    query = _expand_query(query)
    try:
        resp = requests.get(
            _BIOMODELS_SEARCH_URL,
            params={"query": query, "numResults": 10, "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("models", [])
    except Exception:
        return None
    if not results:
        return None
    scored = [
        (r["id"], max(_fuzzy_score(query, r.get("id", "")), _fuzzy_score(query, r.get("name", ""))))
        for r in results
    ]
    best_id, best_score = max(scored, key=lambda x: x[1])
    if best_score < 0.6:
        return None
    return best_id


class ModelManager:
    def __init__(self):
        self.models = {}
        self.current_model_id = None
        self.bounds_data = None
        self.objective = False
        self.sampler = "optgp"

    def validate_model(self, file_path):
        _, errors = validate_sbml_model(file_path)
        for key in ("SBML_FATAL", "COBRA_FATAL", "SBML_ERROR", "SBML_SCHEMA_ERROR", "COBRA_ERROR"):
            if errors.get(key):
                return False
        return True

    def _register_model(self, model, model_id: str) -> str:
        self.models[model_id] = model
        self.current_model_id = model_id
        if model.objective:
            self.objective = True
        return model_id

    def load_model_by_id(self, model_id: str) -> str:
        for prefix in ("BIGG:", "BiGG:", "bigg:", "BioModels:", "biomodels:"):
            if model_id.startswith(prefix):
                model_id = model_id[len(prefix):]
                break
        base_model_id = model_id.split(".")[0]

        # 1. Fast path: direct lookup (exact BiGG or BioModels ID).
        try:
            model = load_model(base_model_id, repositories=[BioModels(), BiGGModels()])
            return self._register_model(model, base_model_id)
        except RuntimeError:
            pass

        # 2. Fuzzy search BiGG.
        bigg_id = _search_bigg(base_model_id)
        if bigg_id:
            try:
                model = load_model(bigg_id, repositories=[BiGGModels()])
                return self._register_model(model, bigg_id)
            except RuntimeError:
                pass

        # 3. Fuzzy search BioModels.
        biomodels_id = _search_biomodels(base_model_id)
        if biomodels_id:
            try:
                model = load_model(biomodels_id, repositories=[BioModels()])
                return self._register_model(model, biomodels_id)
            except RuntimeError:
                pass

        # 4. Nothing found anywhere.
        raise ValueError(
            f"No model found for '{base_model_id}' in BiGG or BioModels. "
            "To get a model for this organism you have two options: "
            "(1) reconstruct one from a PATRIC genome ID using the `build_modelseed_model` tool, or "
            "(2) reconstruct one from a genome FASTA file using the `run_carveme` tool."
        )

    def load_sbml(self, file_path):
        model_id = str(file_path).split("/")[-1].split(".")[0]
        if not self.validate_model(file_path):
            return {"response": f"Error model uploaded is not a valid SBML format."}
        model = read_sbml_model(file_path)
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

    def set_sampler(self, sampler_method):
        if not self.current_model_id:
            return {"response": "No model is currently loaded."}
        self.sampler = sampler_method
        return {"response": "Sampler is successfully set."}



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