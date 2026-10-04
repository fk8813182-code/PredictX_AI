"""Train/compare models, save artifacts + insights. Run: python src/train.py"""
import json, os, sys, datetime as dt, numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.inspection import permutation_importance
from data_loader import *
RS = 42
ROOT = os.path.join(os.path.dirname(__file__), "..")

def preprocessor():
    return ColumnTransformer([("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])

def models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=2, random_state=RS, n_jobs=-1),
        "Gradient Boosting (HistGB)": HistGradientBoostingClassifier(class_weight="balanced", random_state=RS),
    }

def insights(df):
    y = df[TARGET]; out = {}
    out["rows"], out["cols"] = df.shape
    out["missing_values"] = int(df.isna().sum().sum()); out["duplicate_rows"] = int(df.duplicated().sum())
    out["failures"] = int(y.sum()); out["failure_rate_pct"] = round(100 * y.mean(), 2)
    out["feature_means_normal_vs_failed"] = {c: [round(df.loc[y == 0, c].mean(), 2), round(df.loc[y == 1, c].mean(), 2)] for c in NUM}
    out["failure_rate_by_type_pct"] = (df.groupby("Type")[TARGET].mean() * 100).round(2).to_dict()
    out["failure_rate_by_toolwear_quartile_pct"] = (df.groupby(pd.qcut(df["Tool wear [min]"], 4, duplicates="drop"), observed=True)[TARGET].mean() * 100).round(2).rename(index=str).to_dict()
    out["failure_mode_counts"] = {c: int(df[c].sum()) for c in LEAK_COLS if c in df}
    out["correlation_with_failure"] = df[NUM + [TARGET]].corr()[TARGET].drop(TARGET).round(3).to_dict()
    return out

def main():
    df = add_features(load_data())
    ins = insights(df)
    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)   # split BEFORE fitting any transform
    cv = StratifiedKFold(5, shuffle=True, random_state=RS)
    res, fitted = {}, {}
    for name, m in models().items():
        pipe = Pipeline([("pre", preprocessor()), ("clf", m)])
        s = cross_validate(pipe, Xtr, ytr, cv=cv, scoring=["recall", "f1", "roc_auc"])   # VALIDATION (train folds only)
        pipe.fit(Xtr, ytr); p = pipe.predict_proba(Xte)[:, 1]; yh = (p >= 0.5).astype(int)  # TEST (held out)
        fpr, tpr, _ = roc_curve(yte, p); k = max(1, len(fpr) // 100)
        t = {"accuracy": accuracy_score(yte, yh), "precision": precision_score(yte, yh, zero_division=0), "recall": recall_score(yte, yh),
             "f1": f1_score(yte, yh), "roc_auc": roc_auc_score(yte, p)}
        res[name] = {"cv_recall": round(float(s["test_recall"].mean()), 4), "cv_f1": round(float(s["test_f1"].mean()), 4),
                     "cv_roc_auc": round(float(s["test_roc_auc"].mean()), 4), "test": {a: round(float(b), 4) for a, b in t.items()},
                     "confusion_matrix": confusion_matrix(yte, yh).tolist(), "roc": {"fpr": fpr[::k].tolist(), "tpr": tpr[::k].tolist()}}
        fitted[name] = pipe
    # Selection uses VALIDATION only (mean of CV recall, F1, ROC-AUC); the test set stays an unbiased report.
    best = max(res, key=lambda n: np.mean([res[n]["cv_recall"], res[n]["cv_f1"], res[n]["cv_roc_auc"]]))
    pipe = fitted[best]
    pi = permutation_importance(pipe, Xte, yte, scoring="roc_auc", n_repeats=10, random_state=RS)
    imp = dict(sorted({f: round(float(v), 4) for f, v in zip(FEATURES, pi.importances_mean)}.items(), key=lambda kv: -kv[1]))
    ranges = {c: [float(Xtr[c].min()), float(Xtr[c].max()), float(Xtr[c].median())] for c in NUM}
    meta = {"model_name": best, "trained_at": dt.datetime.now().isoformat(timespec="seconds"), "features": FEATURES, "random_state": RS,
            "split": "80/20 stratified; 5-fold stratified CV on train", "dataset": {"rows": ins["rows"], "failures": ins["failures"], "source": "UCI AI4I 2020 (id 601)"},
            "metrics_test": res[best]["test"], "feature_ranges": ranges, "types": sorted(df["Type"].unique().tolist()),
            "risk_thresholds": {"medium": 0.30, "high": 0.60, "note": "Illustrative defaults on UNCALIBRATED probabilities; tune with real maintenance costs."},
            "version": dt.date.today().isoformat()}
    joblib.dump(pipe, os.path.join(ROOT, "models", "final_model.joblib"))
    joblib.dump(pipe.named_steps["pre"], os.path.join(ROOT, "models", "preprocessing.joblib"))
    json.dump(meta, open(os.path.join(ROOT, "models", "model_metadata.json"), "w"), indent=2)
    json.dump({"models": res, "selected": best, "permutation_importance_test_auc_drop": imp, "insights": ins}, open(os.path.join(ROOT, "reports", "model_results.json"), "w"), indent=2)
    with open(os.path.join(ROOT, "reports", "insights.md"), "w") as f:
        f.write("# Dataset insights (auto-generated from the data; associations, not causation)\n\n")
        f.write(f"- {ins['rows']} rows, {ins['failures']} failures ({ins['failure_rate_pct']}%). Missing: {ins['missing_values']}, duplicates: {ins['duplicate_rows']}.\n")
        for c, (a, b) in ins["feature_means_normal_vs_failed"].items(): f.write(f"- {c}: mean {a} (normal) vs {b} (failed) in this dataset.\n")
        f.write(f"- Failure rate by Type (%): {ins['failure_rate_by_type_pct']}\n- Failure rate by tool-wear quartile (%): {ins['failure_rate_by_toolwear_quartile_pct']}\n")
        f.write(f"- Failure-mode counts: {ins['failure_mode_counts']}\n- Correlation with failure: {ins['correlation_with_failure']}\n")
    print("Selected:", best)
    for n, r in res.items(): print(n, "CV(rec,f1,auc)=", r["cv_recall"], r["cv_f1"], r["cv_roc_auc"], "TEST=", r["test"])

if __name__ == "__main__":
    main()
