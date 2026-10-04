import json, os
R = os.path.join(os.path.dirname(__file__), "..", "reports", "model_results.json")
DISCLAIMER = "These are model-level relationships (permutation importance on held-out test data) and should not be interpreted as proof of causation."
def global_importance():
    return json.load(open(R))["permutation_importance_test_auc_drop"]
