"""
Novel: Predict hallucination BEFORE the response is generated.
================================================================
PURPOSE:
  Tests whether 5 cheap signals available BEFORE generation predict hallucination.
  If AUC > 0.70 → you have a novel "pre-emptive hallucination detector."
  This is a stronger contribution than post-hoc detection.

SIGNALS USED (all available at retrieval time, before the model generates):
  1. context_length          — length of retrieved transcript chunks
  2. query_type_base_rate    — historical hallucination rate for this query type
  3. context_depth_score     — completeness score of retrieved context (convo_utils)
  4. is_dutch                — Dutch interviews hallucinate ~33pp more
  5. num_context_chunks      — number of chunks retrieved (more = better coverage)

HOW TO RUN:
  python D:\\RAG_THESIS\\advanced\\hallucination_predictor.py

PREREQUISITES:
  pip install scikit-learn scipy matplotlib
  thesis_hallucination results must exist:
    D:\\thesis_hallucination\\results\\[model]\\01_rag_responses.json
    D:\\thesis_hallucination\\results\\[model]\\02_rq1_annotations.json

OUTPUT:
  - Console: AUC, feature importances, interpretation
  - D:\\RAG_THESIS\\output\\hallucination_predictor_results.json
  - D:\\RAG_THESIS\\output\\predictor_feature_importance.png (if matplotlib available)

USE IN THESIS (Section 5.X — Pre-emptive Hallucination Risk Scoring):
  "We demonstrate that hallucination can be partially predicted before generation
  using retrieval-time signals (AUC = X). Context depth score and query type
  base rate emerge as the strongest predictors (coefficients: Y, Z), suggesting
  that retrieval quality and query difficulty are key confounding factors."
"""

import json
import sys
from pathlib import Path

# Add convo_utils to path
sys.path.insert(0, str(Path(r"D:\thesis_hallucination\src")))

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    print("ERROR: pip install scikit-learn")
    sys.exit(1)

try:
    from convo_utils.depth_scoring import score_completeness
    HAS_DEPTH = True
except ImportError:
    print("WARNING: convo_utils not available. Using context length as depth proxy.")
    HAS_DEPTH = False

OUTPUT_DIR   = Path(r"D:\RAG_THESIS\output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON  = OUTPUT_DIR / "hallucination_predictor_results.json"
OUTPUT_CHART = OUTPUT_DIR / "predictor_feature_importance.png"

THESIS_RESULTS = Path(r"D:\thesis_hallucination\results")
MODELS = ["gpt-4o-mini", "qwen", "mistral"]

# Prior hallucination rates per query type (from your thesis results)
# Update these with your actual numbers after running the pipeline
QUERY_TYPE_BASE_RATES = {
    "sentiment":          0.76,
    "satisfaction":       0.55,
    "challenge":          0.50,
    "improvement":        0.45,
    "timeline":           0.40,
    "speaker_attribution": 0.28,
}


def load_model_data(model: str):
    """Load responses + annotations for one model."""
    rdir = THESIS_RESULTS / model
    resp_file = rdir / "01_rag_responses.json"
    ann_file  = rdir / "02_rq1_annotations.json"

    if not resp_file.exists():
        print(f"  Skipping {model}: {resp_file} not found")
        return [], []

    with open(resp_file, encoding="utf-8") as f:
        responses = json.load(f)

    annotations = []
    if ann_file.exists():
        with open(ann_file, encoding="utf-8") as f:
            annotations = json.load(f)

    return responses, annotations


def build_features(responses, annotations):
    """Build feature matrix X and label vector y."""
    ann_idx = {
        (a.get("interview_id"), a.get("query_type")): a
        for a in annotations
    }

    X, y = [], []
    for r in responses:
        ann = ann_idx.get((r.get("interview_id"), r.get("query_type")))
        if not ann or ann.get("overall_label") not in ("HALLUCINATED", "FAITHFUL"):
            continue

        context = r.get("context", "")
        query_type = r.get("query_type", "unknown")
        language = r.get("language", "en")

        # Feature 1: context length (proxy for retrieval coverage)
        ctx_len = len(context)

        # Feature 2: query type base rate
        base_rate = QUERY_TYPE_BASE_RATES.get(query_type, 0.5)

        # Feature 3: context depth score
        if HAS_DEPTH:
            depth = score_completeness(context[:1000])
        else:
            # Approximate: longer context = more depth
            depth = min(ctx_len / 2000.0, 1.0)

        # Feature 4: is Dutch
        is_dutch = 1.0 if language in ("nl", "dutch", "Nederlands") else 0.0

        # Feature 5: number of sentences in context (proxy for chunk count)
        num_sentences = context.count(".") + context.count("!") + context.count("?")

        X.append([ctx_len / 3000.0,   # normalize
                  base_rate,
                  depth,
                  is_dutch,
                  min(num_sentences / 20.0, 1.0)])

        y.append(1 if ann["overall_label"] == "HALLUCINATED" else 0)

    return X, y


def main():
    print("=" * 60)
    print("HALLUCINATION PREDICTOR — Pre-emptive Risk Scoring")
    print("=" * 60)

    all_X, all_y = [], []
    for model in MODELS:
        print(f"\nLoading {model}...")
        responses, annotations = load_model_data(model)
        X, y = build_features(responses, annotations)
        all_X.extend(X)
        all_y.extend(y)
        print(f"  {len(X)} samples loaded")

    if len(all_X) < 10:
        print("ERROR: Not enough data. Run thesis pipeline first.")
        sys.exit(1)

    print(f"\nTotal samples: {len(all_X)}")
    print(f"Hallucinated: {sum(all_y)} ({sum(all_y)/len(all_y):.1%})")
    print(f"Faithful:     {len(all_y)-sum(all_y)} ({(len(all_y)-sum(all_y))/len(all_y):.1%})")

    # ── train logistic regression ────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(all_X)

    model = LogisticRegression(max_iter=1000, random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(model, X_scaled, all_y, cv=cv, scoring="roc_auc")

    model.fit(X_scaled, all_y)

    feature_names = [
        "context_length",
        "query_type_base_rate",
        "context_depth_score",
        "is_dutch",
        "num_sentences",
    ]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Cross-validated AUC: {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")

    if auc_scores.mean() >= 0.70:
        interp = "STRONG — you can predict hallucination before generation"
    elif auc_scores.mean() >= 0.60:
        interp = "MODERATE — signals partially predict hallucination"
    else:
        interp = "WEAK — hallucination is hard to predict from these signals"
    print(f"Interpretation: {interp}")

    print("\nFeature Importances (logistic regression coefficients):")
    coeffs = model.coef_[0]
    sorted_feats = sorted(zip(feature_names, coeffs), key=lambda x: abs(x[1]), reverse=True)
    for fname, coeff in sorted_feats:
        direction = "↑ more likely HALLUCINATED" if coeff > 0 else "↓ less likely HALLUCINATED"
        bar = "█" * int(abs(coeff) * 10)
        print(f"  {fname:<28} {coeff:+.3f}  {bar}  {direction}")

    top_feature = sorted_feats[0][0]
    top_coeff = sorted_feats[0][1]
    direction_word = "higher" if top_coeff > 0 else "lower"

    print(f"\nUSE IN THESIS:")
    print(f'  "Pre-emptive hallucination prediction achieves AUC = {auc_scores.mean():.2f}')
    print(f"  (95% CI: [{auc_scores.mean()-1.96*auc_scores.std():.2f},")
    print(f"  {auc_scores.mean()+1.96*auc_scores.std():.2f}]) using retrieval-time features.")
    print(f"  {top_feature.replace('_',' ').title()} is the strongest predictor")
    print(f"  (β = {top_coeff:+.3f}): {direction_word} values predict more hallucination,")
    print(f'  suggesting it is a key confounding factor in interview RAG."')

    # ── save results ─────────────────────────────────────────────────────────
    output = {
        "n_samples": len(all_X),
        "hallucination_rate": round(sum(all_y)/len(all_y), 4),
        "cv_auc_mean": round(float(auc_scores.mean()), 4),
        "cv_auc_std": round(float(auc_scores.std()), 4),
        "interpretation": interp,
        "feature_importances": [
            {"feature": f, "coefficient": round(float(c), 4)}
            for f, c in sorted_feats
        ],
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    # ── bar chart ────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        fnames = [item["feature"].replace("_", "\n") for item in output["feature_importances"]]
        fvals  = [item["coefficient"] for item in output["feature_importances"]]
        colors = ["#c0392b" if v > 0 else "#2980b9" for v in fvals]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(range(len(fnames)), fvals, color=colors)
        ax.set_yticks(range(len(fnames)))
        ax.set_yticklabels(fnames, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Logistic Regression Coefficient")
        ax.set_title(f"Hallucination Predictor — Feature Importances (AUC = {auc_scores.mean():.2f})")
        plt.tight_layout()
        plt.savefig(OUTPUT_CHART, dpi=150, bbox_inches="tight")
        print(f"\nChart saved: {OUTPUT_CHART}")
    except ImportError:
        print("(matplotlib not installed — skipping chart)")

    print(f"\nResults saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
