# app_nb23.py -- Clinical XAI Dashboard
# NB23 . Digitalization, AI & XAI in Healthcare . Module 6
#
# Datasets : Cleveland Heart Disease (UCI) . Pima Indians Diabetes (UCI)
# XAI      : SHAP . LIME . Counterfactual . MAPLE . Surrogate Tree . GEMEX
# Run      : streamlit run app_nb23.py

import warnings; warnings.filterwarnings("ignore")
import os, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import shap
import lime
import lime.lime_tabular
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from datetime import datetime

# Try GEMEX
try:
    import gemex
    GEMEX_OK = True
except ImportError:
    GEMEX_OK = False

# ---- Page config -------------------------------------------------------------
st.set_page_config(
    page_title="Clinical XAI Dashboard -- NB23",
    page_icon="[+]",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY   = "#1F3864"; BLUE  = "#2E75B6"; GREEN = "#1F7A5C"
RED    = "#C0392B"; ORANGE= "#D4860B"; PURPLE= "#7B3F9E"
TEAL   = "#117A8B"; BROWN = "#6D4C41"

COLS_CLEVELAND = ["age","sex","cp","trestbps","chol","fbs","restecg",
                  "thalach","exang","oldpeak","slope","ca","thal","target"]
COLS_PIMA      = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
                  "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]

FEAT_LABELS_CLEVELAND = {
    "age":"Age (years)", "sex":"Sex (1=Male)", "cp":"Chest Pain Type",
    "trestbps":"Resting BP (mmHg)", "chol":"Cholesterol (mg/dL)",
    "fbs":"Fasting Blood Sugar>120", "restecg":"Rest ECG",
    "thalach":"Max Heart Rate", "exang":"Exercise Angina",
    "oldpeak":"ST Depression", "slope":"ST Slope",
    "ca":"Major Vessels Coloured", "thal":"Thalassemia"
}
FEAT_LABELS_PIMA = {
    "Pregnancies":"Pregnancies", "Glucose":"Plasma Glucose",
    "BloodPressure":"Blood Pressure (mmHg)", "SkinThickness":"Skin Thickness (mm)",
    "Insulin":"Serum Insulin (uU/mL)", "BMI":"BMI (kg/m2)",
    "DiabetesPedigreeFunction":"Diabetes Pedigree", "Age":"Age (years)"
}

# =============================================================================
# CONCEPT 1 -- @st.cache_resource
# Trains models once. All reruns reuse the cached objects.
# =============================================================================
@st.cache_resource(show_spinner="Training Cleveland Heart model...")
def build_cleveland():
    if os.path.exists("cleveland_heart.csv"):
        df = pd.read_csv("cleveland_heart.csv")
        if df.columns[0] != "age": df.columns = COLS_CLEVELAND
        df = df.replace("?", np.nan).apply(pd.to_numeric, errors="coerce").dropna()
        df["target"] = (df["target"] > 0).astype(int)
    else:
        rng = np.random.default_rng(42); n = 303
        df = pd.DataFrame({
            "age":rng.integers(30,80,n).astype(float),
            "sex":rng.integers(0,2,n).astype(float),
            "cp":rng.integers(0,4,n).astype(float),
            "trestbps":rng.normal(130,20,n),
            "chol":rng.normal(246,50,n),
            "fbs":rng.integers(0,2,n).astype(float),
            "restecg":rng.integers(0,3,n).astype(float),
            "thalach":rng.normal(150,23,n),
            "exang":rng.integers(0,2,n).astype(float),
            "oldpeak":np.abs(rng.normal(1.0,1.2,n)),
            "slope":rng.integers(0,3,n).astype(float),
            "ca":rng.integers(0,4,n).astype(float),
            "thal":rng.integers(1,4,n).astype(float),
        })
        df["target"] = ((df["age"]>55).astype(int)+(df["chol"]>250).astype(int)+(df["trestbps"]>140).astype(int)>1).astype(int)
    feats = [c for c in df.columns if c != "target"]
    X = df[feats].values.astype(np.float32)
    y = df["target"].values.astype(int)
    sc = StandardScaler(); X_sc = sc.fit_transform(X)
    gbm = GradientBoostingClassifier(n_estimators=150,learning_rate=0.1,
                                      max_depth=3,min_samples_leaf=5,random_state=42)
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_cv = cross_val_predict(gbm, X_sc, y, cv=cv, method="predict_proba")[:,1]
    auc  = roc_auc_score(y, y_cv)
    gbm.fit(X_sc, y)
    return dict(df=df, feats=feats, X=X_sc, y=y, gbm=gbm, sc=sc, auc=auc,
                label_names=["No Disease","Heart Disease"],
                feat_labels=FEAT_LABELS_CLEVELAND)

@st.cache_resource(show_spinner="Training Pima Diabetes model...")
def build_pima():
    if os.path.exists("pima_diabetes.csv"):
        raw = pd.read_csv("pima_diabetes.csv", header=0)
        raw = raw[raw.iloc[:,0].astype(str) != raw.columns[0]]
        if raw.columns[0] != "Pregnancies": raw.columns = COLS_PIMA
        df = raw.apply(pd.to_numeric, errors="coerce").dropna()
        df["Outcome"] = df["Outcome"].astype(int)
    else:
        rng = np.random.default_rng(42); n = 768
        df = pd.DataFrame({
            "Pregnancies":rng.integers(0,18,n).astype(float),
            "Glucose":rng.normal(120,32,n),
            "BloodPressure":rng.normal(69,19,n),
            "SkinThickness":rng.normal(20,16,n),
            "Insulin":np.abs(rng.normal(80,115,n)),
            "BMI":rng.normal(32,7,n),
            "DiabetesPedigreeFunction":np.abs(rng.normal(0.47,0.33,n)),
            "Age":rng.integers(21,82,n).astype(float),
        })
        df["Outcome"] = ((df["Glucose"]>140).astype(int)+(df["BMI"]>35).astype(int)+(df["Age"]>50).astype(int)>1).astype(int)
    feats = [c for c in df.columns if c != "Outcome"]
    X = df[feats].values.astype(np.float32)
    y = df["Outcome"].values.astype(int)
    sc = StandardScaler(); X_sc = sc.fit_transform(X)
    gbm = GradientBoostingClassifier(n_estimators=150,learning_rate=0.1,
                                      max_depth=3,min_samples_leaf=5,random_state=42)
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_cv = cross_val_predict(gbm, X_sc, y, cv=cv, method="predict_proba")[:,1]
    auc  = roc_auc_score(y, y_cv)
    gbm.fit(X_sc, y)
    return dict(df=df, feats=feats, X=X_sc, y=y, gbm=gbm, sc=sc, auc=auc,
                label_names=["No Diabetes","Diabetes"],
                feat_labels=FEAT_LABELS_PIMA)

def build_custom(df_raw):
    df = df_raw.apply(pd.to_numeric, errors="coerce").dropna()
    target = df.columns[-1]
    feats  = list(df.columns[:-1])
    X = df[feats].values.astype(np.float32)
    y = (df[target].values > df[target].median()).astype(int)
    sc = StandardScaler(); X_sc = sc.fit_transform(X)
    n_splits = min(5, max(2, int(y.sum())))
    gbm = GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,
                                      max_depth=3,random_state=42)
    cv  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_cv = cross_val_predict(gbm, X_sc, y, cv=cv, method="predict_proba")[:,1]
    auc  = roc_auc_score(y, y_cv)
    gbm.fit(X_sc, y)
    return dict(df=df, feats=feats, X=X_sc, y=y, gbm=gbm, sc=sc, auc=auc,
                label_names=["Low","High"], feat_labels={f:f for f in feats})

# =============================================================================
# SHARED CHART HELPER
# =============================================================================
def make_bar_chart(labels, values, title, xlabel,
                   pos_color=GREEN, neg_color=RED, figsize=(7,4)):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(figsize[0], max(figsize[1], n*0.38)))
    fig.patch.set_facecolor("#0F1923"); ax.set_facecolor("#0F1923")
    colors = [pos_color if v >= 0 else neg_color for v in values]
    idx    = np.argsort(np.abs(values))
    ax.barh([labels[i] for i in idx], [values[i] for i in idx],
             color=[colors[i] for i in idx], edgecolor="none")
    ax.axvline(0, color="white", lw=0.8)
    ax.set_xlabel(xlabel, color="white")
    ax.set_title(title, color="white", fontweight="bold", fontsize=10)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#444"); ax.spines["left"].set_color("#444")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    plt.tight_layout()
    return fig

# =============================================================================
# XAI METHOD 1 -- SHAP (NB2)
# Tree-based Shapley values -- gold standard for tabular GBM
# =============================================================================
def run_shap(backend, idx):
    gbm=backend["gbm"]; X=backend["X"]
    feats=backend["feats"]; flab=backend["feat_labels"]
    expl = shap.TreeExplainer(gbm)
    sv   = expl.shap_values(X[idx:idx+1])[0]
    labels = [flab.get(f,f) for f in feats]
    fig = make_bar_chart(labels, sv,
                          "SHAP -- Feature Attributions",
                          "SHAP value (impact on prediction)")
    desc = (
        "SHAP (SHapley Additive exPlanations) uses game-theory Shapley values to "
        "fairly distribute the prediction gap among features. "
        "Green bars push the prediction toward the positive class; "
        "red bars push it toward the negative class. "
        "Values sum exactly to the total prediction shift from the base rate."
    )
    return fig, dict(zip(feats, sv.tolist())), desc

# =============================================================================
# XAI METHOD 2 -- LIME (NB3)
# Local Interpretable Model-agnostic Explanations
# =============================================================================
def run_lime(backend, idx):
    gbm=backend["gbm"]; X=backend["X"]
    feats=backend["feats"]; flab=backend["feat_labels"]
    def predict_fn(arr): return gbm.predict_proba(arr)
    expl = lime.lime_tabular.LimeTabularExplainer(
        X, feature_names=feats, class_names=backend["label_names"],
        mode="classification", random_state=42)
    exp  = expl.explain_instance(X[idx], predict_fn,
                                  num_features=len(feats), num_samples=500)
    lime_dict = dict(exp.as_list())
    sv = np.zeros(len(feats))
    for i,f in enumerate(feats):
        for k,v in lime_dict.items():
            if f in k: sv[i]=v; break
    labels = [flab.get(f,f) for f in feats]
    fig = make_bar_chart(labels, sv,
                          "LIME -- Local Feature Weights",
                          "LIME weight (local attribution)")
    desc = (
        "LIME perturbs this patient's features and fits a local linear model nearby. "
        "Weights show which features most influenced the prediction in this "
        "specific neighbourhood. Note: LIME can be unstable -- running it twice "
        "may give slightly different results (unlike SHAP which is deterministic)."
    )
    return fig, dict(zip(feats, sv.tolist())), desc

# =============================================================================
# XAI METHOD 3 -- MAPLE (NB3)
# More stable local explanations via Random Forest leaf co-membership
# =============================================================================
def run_maple(backend, idx):
    gbm=backend["gbm"]; X=backend["X"]
    feats=backend["feats"]; flab=backend["feat_labels"]
    y=backend["y"]
    # MAPLE: fit RF, use leaf co-membership to weight training points,
    # then fit a local Ridge regression
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y.astype(float))
    leaves = rf.apply(X)              # (n_patients, n_trees)
    x_leaves = rf.apply(X[idx:idx+1]) # (1, n_trees)
    # Weight = fraction of trees where patient shares a leaf with x
    weights = (leaves == x_leaves).mean(axis=1).astype(np.float64)
    weights /= (weights.sum() + 1e-10)
    # Fit weighted Ridge on raw features
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, gbm.predict_proba(X)[:,1], sample_weight=weights)
    coefs = ridge.coef_
    labels = [flab.get(f,f) for f in feats]
    fig = make_bar_chart(labels, coefs,
                          "MAPLE -- Local Ridge Coefficients",
                          "Coefficient (local importance)",
                          pos_color=TEAL, neg_color=BROWN)
    desc = (
        "MAPLE (Model-Agnostic Pseudo-Local Explanations) uses Random Forest "
        "leaf co-membership to identify the most similar training patients, "
        "then fits a local weighted Ridge regression. It is more stable than LIME "
        "because the neighbourhood is defined by the model's own structure, "
        "not random perturbations."
    )
    return fig, dict(zip(feats, coefs.tolist())), desc

# =============================================================================
# XAI METHOD 4 -- Surrogate Decision Tree (NB4)
# Global interpretable surrogate -- human-readable rules
# =============================================================================
def run_surrogate_tree(backend, idx):
    gbm=backend["gbm"]; X=backend["X"]
    feats=backend["feats"]; flab=backend["feat_labels"]
    # Train a shallow decision tree to mimic the GBM globally
    y_soft = gbm.predict_proba(X)[:,1]
    y_hard = (y_soft > 0.5).astype(int)
    tree   = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42)
    tree.fit(X, y_hard)
    fidelity = (tree.predict(X) == y_hard).mean()
    # Visualise the tree
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0F1923"); ax.set_facecolor("#0F1923")
    plot_tree(tree,
              feature_names=[flab.get(f,f) for f in feats],
              class_names=backend["label_names"],
              filled=True, rounded=True, fontsize=7,
              ax=ax,
              proportion=False,
              impurity=False)
    ax.set_title(
        f"Surrogate Decision Tree (depth=4, fidelity={fidelity:.1%})\n"
        f"Approximates the GBM globally with human-readable IF-THEN rules",
        color="white", fontweight="bold", fontsize=10, pad=10)
    fig.patch.set_facecolor("#0F1923")
    plt.tight_layout()
    # Also get the text rule for this patient
    path  = tree.decision_path(X[idx:idx+1])
    rules = export_text(tree,
                        feature_names=[flab.get(f,f) for f in feats])
    # Find which path the current patient takes
    node_ids = path.indices
    all_rules = rules.split("\n")
    patient_pred = backend["label_names"][int(tree.predict(X[idx:idx+1])[0])]
    desc = (
        f"A shallow Decision Tree (depth=4) trained to mimic the GBM -- "
        f"fidelity {fidelity:.1%}. "
        f"It produces human-readable IF-THEN rules that approximate the black-box model globally. "
        f"For Patient {idx}, the surrogate predicts: {patient_pred}. "
        f"Clinical use: the tree rules can be printed on a clinical decision aid card."
    )
    vals = dict(zip(feats, tree.feature_importances_.tolist()))
    return fig, vals, desc

# =============================================================================
# XAI METHOD 5 -- Counterfactual (NB4)
# Nearest opposite patient -- actionable clinical advice
# =============================================================================
def run_counterfactual(backend, idx):
    gbm=backend["gbm"]; X=backend["X"]
    y=backend["y"]; feats=backend["feats"]
    flab=backend["feat_labels"]
    x0=X[idx]; pred0=gbm.predict([x0])[0]
    opposite = np.where(y != pred0)[0]
    if len(opposite) == 0: return None, {}, "No counterfactual found."
    cf_idx = opposite[np.argmin(np.linalg.norm(X[opposite]-x0, axis=1))]
    delta  = X[cf_idx] - x0
    labels = [flab.get(f,f) for f in feats]
    top8   = np.argsort(np.abs(delta))[-8:]
    fig    = make_bar_chart(
        [labels[i] for i in top8], [delta[i] for i in top8],
        f"Counterfactual -- What would flip the prediction?\n"
        f"({backend['label_names'][pred0]} -> {backend['label_names'][1-pred0]})",
        "Feature change (CF minus original, standardised)",
        pos_color=ORANGE, neg_color=PURPLE)
    desc = (
        "The Counterfactual is the nearest patient in the dataset who received "
        "the opposite prediction. The bars show which features differ and by how much. "
        "Clinically: this answers 'what would need to change for this patient "
        "to be classified differently?' -- the most actionable form of explanation."
    )
    return fig, dict(zip(feats, delta.tolist())), desc

# =============================================================================
# XAI METHOD 6 -- GEMEX (NB22)
# Geodesic Entropic Manifold Explainability
# Information-geometric arc-length from the reference manifold
# =============================================================================
def run_gemex(backend, idx):
    gbm=backend["gbm"]; X=backend["X"]
    feats=backend["feats"]; flab=backend["feat_labels"]
    y=backend["y"]

    if not GEMEX_OK:
        return None, {}, "GEMEX not installed. Run: pip install gemex"

    # Build numpy head from GBM
    # GEMEX needs a predict function on embeddings
    # We use GBM leaf node counts as embeddings
    X_leaves_3d = gbm.apply(X)  # GradientBoosting.apply returns (n, n_estimators, 1)
    X_leaves = X_leaves_3d[:, :, 0].astype(np.float32)  # reshape to 2D (n, n_estimators)
    from sklearn.preprocessing import normalize
    X_emb = normalize(X_leaves, norm="l2")

    # Reference: negative class patients (label=0)
    ref_idx = np.where(y == 0)[0][:20]
    X_ref   = X_emb[ref_idx]

    # Simple predict function on embeddings via Ridge
    ridge_emb = Ridge(alpha=1.0)
    ridge_emb.fit(X_emb, gbm.predict_proba(X)[:,1])
    def predict_fn(arr):
        arr = np.atleast_2d(arr)
        scores = ridge_emb.predict(arr)
        scores = np.clip(scores, 0, 1)
        return np.column_stack([1-scores, scores])

    try:
        gx = gemex.Explainer(
            predict_fn,
            data_type="tabular",
            feature_names=[f"leaf_{i}" for i in range(X_emb.shape[1])],
            class_names=backend["label_names"],
            config=gemex.GemexConfig(
                n_geodesic_steps=6,
                n_reference_samples=min(10, len(X_ref)),
                interaction_order=1,
                verbose=False))
        r   = gx.explain(X_emb[idx], X_reference=X_ref)
        arc = float(r.geodesic_lengths[-1])
        gsf = np.abs(r.gsf_scores)

        # Map GSF scores back to original features via correlation
        feat_importance = np.zeros(len(feats))
        for fi in range(len(feats)):
            col = X[:, fi]
            # Correlate each feature with GBM leaf activations
            leaf_corrs = np.abs([np.corrcoef(col, X_leaves[:,li])[0,1]
                                  for li in range(X_leaves.shape[1])])
            # Weight by GSF
            n_gsf = min(len(gsf), len(leaf_corrs))
            feat_importance[fi] = np.dot(leaf_corrs[:n_gsf], gsf[:n_gsf] / (gsf[:n_gsf].sum()+1e-10))

        feat_importance = feat_importance / (feat_importance.max()+1e-10)

        labels = [flab.get(f,f) for f in feats]
        fig = make_bar_chart(labels, feat_importance,
                              f"GEMEX -- Geodesic Feature Sensitivity\nArc-length from reference manifold: {arc:.4f}",
                              "Geodesic sensitivity (0=low, 1=high)",
                              pos_color=PURPLE, neg_color=TEAL)
        desc = (
            f"GEMEX (Geodesic Entropic Manifold Explainability) measures how far "
            f"this patient's model embedding lies from the reference manifold "
            f"(negative class patients) in Fisher-Rao information space. "
            f"Arc-length = {arc:.4f} -- "
            f"{'high: patient is structurally distant from the reference group' if arc > 0.3 else 'low: patient is close to the reference group'}. "
            f"Bars show which original features contribute most to this geodesic deviation."
        )
        return fig, dict(zip(feats, feat_importance.tolist())), desc

    except Exception as e:
        return None, {}, f"GEMEX error: {str(e)}"

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0); return buf.read()

# =============================================================================
# METHOD REGISTRY -- add new methods here to extend the dashboard
# =============================================================================
XAI_METHODS = {
    "SHAP":             (run_shap,           GREEN,  "NB2", "Feature attribution (Shapley values)"),
    "LIME":             (run_lime,           BLUE,   "NB3", "Local linear surrogate"),
    "MAPLE":            (run_maple,          TEAL,   "NB3", "Stable local surrogate (RF co-membership)"),
    "Counterfactual":   (run_counterfactual, ORANGE, "NB4", "Nearest opposite patient"),
    "Surrogate Tree":   (run_surrogate_tree, BROWN,  "NB4", "Global IF-THEN rules (depth-4 tree)"),
    "GEMEX":            (run_gemex,          PURPLE, "NB22","Geodesic information-geometric deviation"),
}

# =============================================================================
# CONCEPT 2 -- st.sidebar
# =============================================================================
with st.sidebar:
    st.markdown(
        f'<div style="background:{NAVY};padding:14px;border-radius:8px;margin-bottom:12px">'
        f'<h3 style="color:#E8E8F0;margin:0;font-size:1em">Clinical XAI Dashboard</h3>'
        f'<p style="color:#9999BB;margin:4px 0 0;font-size:.78em">NB23 . Module 6 . 6 XAI Methods</p>'
        f'</div>', unsafe_allow_html=True)

    dataset_choice = st.selectbox(
        "Dataset",
        ["Cleveland Heart Disease", "Pima Indians Diabetes", "Upload CSV"],
        help="Choose a built-in dataset or upload your own CSV")

    uploaded = None
    if dataset_choice == "Upload CSV":
        uploaded = st.file_uploader(
            "Upload CSV (last column = target)", type=["csv"],
            help="Numeric CSV. Last column treated as target variable.")

    st.markdown("---")
    xai_method = st.radio(
        "XAI Method",
        list(XAI_METHODS.keys()),
        help="\n".join([f"{k}: {v[3]}" for k,v in XAI_METHODS.items()]))

    method_color = XAI_METHODS[xai_method][1]
    method_nb    = XAI_METHODS[xai_method][2]
    method_desc  = XAI_METHODS[xai_method][3]

    st.markdown(
        f'<div style="background:#131326;padding:8px;border-radius:6px;'
        f'border-left:3px solid {method_color};margin-top:6px">'
        f'<span style="color:{method_color};font-size:.80em;font-weight:bold">{xai_method}</span>'
        f'<br><span style="color:#9999BB;font-size:.74em">{method_desc}</span>'
        f'<br><span style="color:#555;font-size:.70em">Introduced in {method_nb}</span>'
        f'</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        f'<div style="background:#131326;padding:8px;border-radius:6px;font-size:.74em;color:#9999BB">'
        f'<b style="color:#E8E8F0">All 6 XAI Methods</b><br>'
        + "".join([
            f'<span style="color:{v[1]}">. {k}</span> ({v[3][:25]}...)<br>'
            for k,v in XAI_METHODS.items()])
        + f'</div>', unsafe_allow_html=True)

# ---- Load backend ------------------------------------------------------------
if dataset_choice == "Cleveland Heart Disease":
    backend = build_cleveland(); ds_name = "Cleveland Heart Disease"; ds_color = RED
elif dataset_choice == "Pima Indians Diabetes":
    backend = build_pima(); ds_name = "Pima Indians Diabetes"; ds_color = BLUE
else:
    if uploaded is None:
        st.info("Upload a CSV file in the sidebar to continue."); st.stop()
    df_raw  = pd.read_csv(uploaded)
    backend = build_custom(df_raw)
    ds_name = uploaded.name; ds_color = ORANGE

n_patients = len(backend["df"])

with st.sidebar:
    patient_idx = st.slider("Patient index", 0, n_patients-1, 0,
                             help="Select a patient from the dataset")
    st.caption(f"Dataset: {n_patients} patients . AUC: {backend['auc']:.3f}")

# ---- Header ------------------------------------------------------------------
st.markdown(
    f'<div style="background:{NAVY};padding:16px 20px;border-radius:8px;margin-bottom:16px">'
    f'<h2 style="color:#FFFFFF;margin:0;font-size:1.3em">Clinical XAI Dashboard</h2>'
    f'<p style="color:#A8C8E8;margin:4px 0 0;font-size:.88em">'
    f'Dataset: <b>{ds_name}</b> . '
    f'Model AUC: <b>{backend["auc"]:.3f}</b> . '
    f'XAI Method: <b style="color:{method_color}">{xai_method}</b>'
    f'</p></div>', unsafe_allow_html=True)

# =============================================================================
# CONCEPT 3 -- st.tabs
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Patient View",
    "Compare XAI Methods",
    "Dataset Explorer",
    "Model Info"
])

# ---- TAB 1: Patient View -----------------------------------------------------
with tab1:
    row        = backend["df"].iloc[patient_idx]
    x_raw      = backend["X"][patient_idx]
    prob       = backend["gbm"].predict_proba([x_raw])[0]
    pred_class = int(np.argmax(prob))
    confidence = float(prob[pred_class])
    true_label = int(backend["y"][patient_idx])
    label_names= backend["label_names"]
    pred_color = RED if pred_class == 1 else GREEN
    correct    = "Correct" if pred_class == true_label else "Incorrect"

    st.markdown(
        f'<div style="background:{pred_color}22;border:2px solid {pred_color};'
        f'border-radius:8px;padding:14px 18px;margin-bottom:16px">'
        f'<span style="color:{pred_color};font-size:1.2em;font-weight:bold">'
        f'Prediction: {label_names[pred_class]}</span>'
        f'&nbsp;&nbsp;&nbsp;'
        f'<span style="color:#E8E8F0">Confidence: {confidence:.1%}</span>'
        f'&nbsp;&nbsp;&nbsp;'
        f'<span style="color:#9999BB">True label: {label_names[true_label]} . {correct}</span>'
        f'</div>', unsafe_allow_html=True)

    col_feat, col_xai = st.columns([1, 1.6])

    with col_feat:
        st.subheader(f"Patient {patient_idx} -- Features")
        flab = backend["feat_labels"]
        feat_df = pd.DataFrame({
            "Feature": [flab.get(f,f) for f in backend["feats"]],
            "Value":   [f"{row[f]:.2f}" for f in backend["feats"]]
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    with col_xai:
        st.subheader(f"{xai_method} Explanation")
        with st.spinner(f"Computing {xai_method}..."):
            run_fn = XAI_METHODS[xai_method][0]
            fig, vals, explanation_desc = run_fn(backend, patient_idx)

        if fig is not None:
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        else:
            st.warning(explanation_desc)

        if explanation_desc:
            st.markdown(
                f'<div style="background:#131326;padding:10px;border-radius:6px;'
                f'border-left:3px solid {method_color};margin-top:8px">'
                f'<p style="color:#9999BB;font-size:.80em;margin:0">'
                f'<b style="color:{method_color}">How to read this:</b> {explanation_desc}'
                f'</p></div>', unsafe_allow_html=True)

    # CONCEPT 4 -- st.session_state
    if "history" not in st.session_state:
        st.session_state.history = []
    if fig is not None:
        entry = {"time": datetime.now().strftime("%H:%M:%S"),
                 "patient": patient_idx, "method": xai_method,
                 "prediction": label_names[pred_class],
                 "confidence": f"{confidence:.1%}", "dataset": ds_name}
        if not st.session_state.history or \
           (st.session_state.history[-1]["patient"] != patient_idx or
            st.session_state.history[-1]["method"] != xai_method):
            st.session_state.history.append(entry)
            if len(st.session_state.history) > 5:
                st.session_state.history.pop(0)

    if st.session_state.history:
        st.markdown("---")
        st.subheader("Session History (last 5 explanations)")
        st.dataframe(pd.DataFrame(st.session_state.history[::-1]),
                     use_container_width=True, hide_index=True)

    # CONCEPT 5 -- st.download_button
    st.markdown("---")
    if fig is not None and vals:
        flab = backend["feat_labels"]
        report = [
            "Clinical XAI Report -- NB23",
            f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Dataset   : {ds_name}",
            f"Patient   : {patient_idx}",
            f"Method    : {xai_method}",
            f"Prediction: {label_names[pred_class]} ({confidence:.1%})",
            f"True label: {label_names[true_label]}",
            f"Model AUC : {backend['auc']:.3f}",
            "",
            f"Method description: {explanation_desc}",
            "",
            "Feature scores (sorted by magnitude):",
        ]
        for f,v in sorted(vals.items(), key=lambda x: abs(x[1]), reverse=True):
            report.append(f"  {flab.get(f,f):35s}: {v:+.4f}")
        report_text = "\n".join(report)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="Download XAI Report (.txt)", data=report_text,
                file_name=f"xai_p{patient_idx}_{xai_method.lower().replace(' ','_')}.txt",
                mime="text/plain")
        with col_dl2:
            st.download_button(
                label="Download Plot (.png)", data=fig_to_bytes(fig),
                file_name=f"xai_p{patient_idx}_{xai_method.lower().replace(' ','_')}.png",
                mime="image/png")

# ---- TAB 2: Compare XAI Methods (NEW) ----------------------------------------
with tab2:
    st.subheader(f"Compare All 6 XAI Methods -- Patient {patient_idx}")
    st.markdown(
        f'<p style="color:#9999BB;font-size:.85em">'
        f'Run all six methods on Patient {patient_idx} and see how they agree or disagree. '
        f'Agreement across methods increases clinical confidence; disagreement signals '
        f'areas where the model behaviour is complex or ambiguous.</p>',
        unsafe_allow_html=True)

    run_comparison = st.button("Run All 6 Methods on this Patient", type="primary")

    if run_comparison:
        results = {}
        prog = st.progress(0)
        status = st.empty()
        for i, (mname, (mfn, mcol, mnb, mdesc)) in enumerate(XAI_METHODS.items()):
            status.text(f"Running {mname}...")
            try:
                mfig, mvals, mdescription = mfn(backend, patient_idx)
                results[mname] = (mfig, mvals, mcol, mdescription)
            except Exception as e:
                results[mname] = (None, {}, mcol, f"Error: {str(e)}")
            prog.progress((i+1)/len(XAI_METHODS))
        status.text("All methods complete.")

        # Show top feature per method in a comparison table
        feats  = backend["feats"]
        flab   = backend["feat_labels"]
        comp_rows = []
        for mname, (mfig, mvals, mcol, mdescription) in results.items():
            if mvals:
                top_f = max(mvals, key=lambda k: abs(mvals[k]))
                top_v = mvals[top_f]
                comp_rows.append({
                    "Method":       mname,
                    "Notebook":     XAI_METHODS[mname][2],
                    "Top Feature":  flab.get(top_f, top_f),
                    "Score":        f"{top_v:+.4f}",
                    "Direction":    "Risk+" if top_v > 0 else "Risk-",
                })
        if comp_rows:
            st.markdown("**Top feature identified by each method:**")
            comp_df = pd.DataFrame(comp_rows)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            # Check agreement
            top_feats = [r["Top Feature"] for r in comp_rows]
            majority  = max(set(top_feats), key=top_feats.count)
            agree_n   = top_feats.count(majority)
            st.info(
                f"Majority agreement: **{majority}** identified as top feature "
                f"by {agree_n}/{len(top_feats)} methods. "
                f"{'Strong agreement -- high confidence in this feature.' if agree_n >= 4 else 'Mixed agreement -- model behaviour is complex for this patient.'}"
            )

        st.markdown("---")
        st.markdown("**Explanation plots for all methods:**")
        cols = st.columns(2)
        for i, (mname, (mfig, mvals, mcol, mdescription)) in enumerate(results.items()):
            with cols[i % 2]:
                st.markdown(
                    f'<div style="border:1px solid {mcol};border-radius:6px;'
                    f'padding:8px;margin-bottom:8px">'
                    f'<b style="color:{mcol}">{mname}</b>'
                    f'<span style="color:#555;font-size:.72em"> . {XAI_METHODS[mname][2]}</span>'
                    f'</div>', unsafe_allow_html=True)
                if mfig is not None:
                    st.pyplot(mfig, use_container_width=True); plt.close(mfig)
                else:
                    st.warning(mdescription)

# ---- TAB 3: Dataset Explorer -------------------------------------------------
with tab3:
    df   = backend["df"]
    feats= backend["feats"]
    tgt  = [c for c in df.columns if c not in feats][0]
    flab = backend["feat_labels"]

    st.subheader(f"Dataset: {ds_name}")
    c1,c2,c3 = st.columns(3)
    c1.metric("Patients", len(df))
    c2.metric("Features", len(feats))
    c3.metric("Positive class rate", f"{df[tgt].mean():.1%}")

    st.dataframe(df.head(50), use_container_width=True)
    st.markdown("---")

    col_dist, col_stats = st.columns(2)
    with col_dist:
        st.subheader("Class Distribution")
        counts = df[tgt].value_counts().sort_index()
        fig_d, ax_d = plt.subplots(figsize=(4,3))
        fig_d.patch.set_facecolor("#0F1923"); ax_d.set_facecolor("#0F1923")
        bars = ax_d.bar([backend["label_names"][int(i)] for i in counts.index],
                         counts.values, color=[GREEN, RED][:len(counts)])
        ax_d.bar_label(bars, color="white")
        ax_d.tick_params(colors="white")
        ax_d.set_title("Class Distribution", color="white", fontweight="bold")
        for sp in ["top","right"]: ax_d.spines[sp].set_visible(False)
        ax_d.spines["bottom"].set_color("#444"); ax_d.spines["left"].set_color("#444")
        st.pyplot(fig_d, use_container_width=True); plt.close(fig_d)
    with col_stats:
        st.subheader("Feature Statistics")
        st.dataframe(df[feats].describe().T.round(2), use_container_width=True)

# ---- TAB 4: Model Info -------------------------------------------------------
with tab4:
    st.subheader("Model Performance")
    c1,c2,c3 = st.columns(3)
    c1.metric("Model", "Gradient Boosting")
    c2.metric("CV AUC", f"{backend['auc']:.3f}")
    c3.metric("Patients", len(backend["df"]))

    st.markdown("---")
    st.subheader("GBM Feature Importance")
    fi   = backend["gbm"].feature_importances_
    feats= backend["feats"]; flab = backend["feat_labels"]
    fig_fi, ax_fi = plt.subplots(figsize=(7,4))
    fig_fi.patch.set_facecolor("#0F1923"); ax_fi.set_facecolor("#0F1923")
    idx = np.argsort(fi)
    ax_fi.barh([flab.get(feats[i],feats[i]) for i in idx], fi[idx],
                color=BLUE, edgecolor="none")
    ax_fi.set_xlabel("Importance", color="white")
    ax_fi.set_title("GBM Feature Importance", color="white", fontweight="bold")
    ax_fi.tick_params(colors="white")
    ax_fi.spines["bottom"].set_color("#444"); ax_fi.spines["left"].set_color("#444")
    for sp in ["top","right"]: ax_fi.spines[sp].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_fi, use_container_width=True); plt.close(fig_fi)

    st.markdown("---")
    st.subheader("XAI Method Reference Guide")
    method_info = [
        (GREEN,  "SHAP",           "NB2", "Shapley values",
         "Theoretically grounded (4 axioms). Deterministic. Gold standard for tabular GBM."),
        (BLUE,   "LIME",           "NB3", "Local linear surrogate",
         "Fast and model-agnostic. Can be unstable across runs. Good for quick prototyping."),
        (TEAL,   "MAPLE",          "NB3", "RF leaf co-membership",
         "More stable than LIME. Uses model structure for neighbourhood. Better for audits."),
        (ORANGE, "Counterfactual", "NB4", "Nearest opposite patient",
         "Most actionable. Answers: what would need to change? Best for patient communication."),
        (BROWN,  "Surrogate Tree", "NB4", "Global IF-THEN rules",
         "Human-readable globally. Can be printed on a clinical decision card."),
        (PURPLE, "GEMEX",          "NB22","Fisher-Rao geodesic",
         "Information-geometric. Measures structural deviation from reference group."),
    ]
    cols = st.columns(3)
    for i, (col, name, nb_ref, mtype, desc) in enumerate(method_info):
        with cols[i % 3]:
            st.markdown(
                f'<div style="background:#131326;padding:12px;border-radius:8px;'
                f'border:2px solid {col};margin-bottom:10px">'
                f'<b style="color:{col}">{name}</b>'
                f'<span style="color:#555;font-size:.72em"> . {nb_ref} . {mtype}</span>'
                f'<p style="color:#9999BB;font-size:.80em;margin:6px 0 0">{desc}</p>'
                f'</div>', unsafe_allow_html=True)

    if not GEMEX_OK:
        st.warning("GEMEX not installed. Install with: pip install gemex")
