import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
import os
from datasets import load_from_disk
import joblib
import torch
import torch.nn.functional as F
import io

# ---------------------------------------------------
#  NASA-Style Page Settings (Custom Earthy Theme)
# ---------------------------------------------------
st.set_page_config(
    page_title="Galaxy Morphology Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Theme: background #404e3b, accents #6c8480 & #bac8b1
st.markdown("""
<style>
body { background-color: #404e3b; color: #f2f3f1; }
.stApp { background-color: #404e3b; }
.stSidebar { background-color: #353f31 !important; }
.stPlotlyChart { background-color: #404e3b !important; }
.stImage { border-radius: 10px; }

/* ---------------- DROPDOWNS & NUMBER INPUTS ---------------- */
/* All selectboxes (dropdowns) */
div[data-baseweb="select"] > div,
div[data-testid="stSelectbox"] > div[role="combobox"],
div[data-baseweb="select"] > div > div {
    background-color: #7b9669 !important;
    color: #f2f3f1 !important;
    border: 1px solid #6c8480 !important;
    border-radius: 4px;
}

/* Number inputs */
div[data-testid="stNumberInput"] input {
    background-color: #7b9669 !important;
    color: #f2f3f1 !important;
    border: 1px solid #6c8480 !important;
}

/* Dropdown arrows and icons */
div[data-baseweb="select"] svg,
div[data-testid="stSelectbox"] svg {
    fill: #f2f3f1 !important;
}

/* Dropdown menu lists */
ul[role="listbox"] {
    background-color: #7b9669 !important;
    border: 1px solid #6c8480 !important;
}
ul[role="listbox"] li {
    color: #f2f3f1 !important;
}
ul[role="listbox"] li:hover {
    background-color: #6c8480 !important;
}

/* ---------------- TABLE STYLE ---------------- */
.stDataFrame { background-color: #404e3b !important; }
.stDataFrame thead th {
    background-color: #bac8b1 !important;
    color: #000000 !important;
    font-weight: bold !important;
    border-bottom: 2px solid #6c8480 !important;
}
.stDataFrame tbody td {
    background-color: #e6e6e6 !important;
    color: #000000 !important;
    border: 1px solid #6c8480 !important;
}
.stDataFrame tbody tr:hover td {
    background-color: #bac8b1 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-thumb { background: #6c8480; border-radius: 5px; }
::-webkit-scrollbar-track { background: #353f31; }
</style>
""", unsafe_allow_html=True)

st.title("🌌 **Galaxy Morphology Dashboard**")
st.caption("Machine Learning • CNN Classification • Clustering • Anomalies")

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE), "gz2_prepared")
UNSUP_PATH = os.path.join(os.path.dirname(BASE), "outputs/unsupervised/unsupervised_results.csv")

IMG_ANOM = os.path.join(os.path.dirname(BASE), "outputs/unsupervised/anomaly_detection.png")
IMG_TSNE = os.path.join(os.path.dirname(BASE), "outputs/unsupervised/tsne_visualization.png")

# ML models
RF_PATH = os.path.join(os.path.dirname(BASE), "outputs/models/random_forest.pkl")
LR_PATH = os.path.join(os.path.dirname(BASE), "outputs/models/logistic_regression.pkl")
SCALER_PATH = os.path.join(os.path.dirname(BASE), "outputs/models/scaler.pkl")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_resource
def load_gz2():
    return load_from_disk(DATA_PATH)

@st.cache_resource
def load_unsupervised_results():
    return pd.read_csv(UNSUP_PATH)

ds = load_gz2()
unsup_df = load_unsupervised_results()

# Load ML models
rf = joblib.load(RF_PATH)
lr = joblib.load(LR_PATH)
scaler = joblib.load(SCALER_PATH)

# ---------------------------------------------------
# Dummy CNN Predictor
# ---------------------------------------------------
def cnn_predict_dummy():
    return np.random.dirichlet([1, 1, 1])

# ---------------------------------------------------
# Image Loader
# ---------------------------------------------------
def get_img(idx):
    img = ds[int(idx)]["image"]
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, dict) and "bytes" in img:
        return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    return None

# ---------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------
st.sidebar.title("📍 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Galaxy Viewer", "ML + CNN Predictions", "Anomalies", "Clusters", "t-SNE Map"]
)

# ---------------------------------------------------
# PAGE 1 — Galaxy Viewer
# ---------------------------------------------------
if page == "Galaxy Viewer":
    st.header("🔭 Explore a Galaxy")
    st.markdown("""
    **This page includes:**  
    - Galaxy Image  
    - Key morphology fractions  
    - Simplified clean bar chart  
    """)
    idx = st.number_input("Galaxy Index", min_value=0, max_value=len(ds)-1, value=0)
    row = ds[int(idx)]
    img = get_img(idx)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("🖼 Galaxy Image")
        st.image(img, use_container_width=True)
    with col2:
        st.subheader("📊 Main Morphology Fractions")
        main_cols = ["smooth_fraction","disk_fraction","spiral_fraction","edge_on_fraction","odd_fraction"]
        morph = {c: row.get(c, 0) for c in main_cols}
        df_m = pd.DataFrame({
            "Feature": [c.replace("_fraction", "").replace("_"," ").title() for c in main_cols],
            "Fraction": list(morph.values())
        })
        fig = px.bar(
            df_m, x="Feature", y="Fraction", range_y=[0,1],
            title="Simplified Morphology Overview",
            color_discrete_sequence=["#6c8480", "#bac8b1"]
        )
        fig.update_layout(paper_bgcolor="#404e3b", plot_bgcolor="#404e3b",
                          font=dict(color="#f2f3f1"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# PAGE 2 — ML + CNN Predictions
# ---------------------------------------------------
elif page == "ML + CNN Predictions":
    st.header("🛰 Machine Learning & CNN Predictions")
    st.markdown("ML models use morphology features to classify a galaxy. CNN predictions shown are dummy (randomized).")
    idx = st.number_input("Galaxy Index for Prediction", min_value=0, max_value=len(ds)-1)
    img = get_img(idx)
    row = ds[int(idx)]
    st.image(img, width=250)

    frac_cols = [c for c in row.keys() if c.endswith("_fraction")]
    X = np.array([0.0 if pd.isna(row[c]) else row[c] for c in frac_cols]).reshape(1,-1)
    X_scaled = scaler.transform(X)
    pred_rf = rf.predict_proba(X_scaled)[0]
    pred_lr = lr.predict_proba(X_scaled)[0]
    pred_cnn = cnn_predict_dummy()
    df_pred = pd.DataFrame({"Class":["Smooth","Disk","Spiral"], "RF":pred_rf, "LR":pred_lr, "CNN (Dummy)":pred_cnn})

    st.subheader("📊 Prediction Probabilities")
    def render_table(df):
        header_color = "#bac8b1"
        cell_color = "#e6e6e6"
        html = f'<table style="border-collapse: collapse; width: 100%;">'
        html += "<thead><tr>"
        for col in df.columns:
            html += f'<th style="background-color:{header_color}; color:#000; padding:6px; border:1px solid #6c8480;">{col}</th>'
        html += "</tr></thead><tbody>"
        for _, row in df.iterrows():
            html += "<tr>"
            for val in row:
                display_val = f"{val:.2f}" if isinstance(val,(float,np.floating)) else str(val)
                html += f'<td style="background-color:{cell_color}; color:#000; padding:6px; border:1px solid #6c8480;">{display_val}</td>'
            html += "</tr>"
        html += "</tbody></table>"
        return html
    st.markdown(render_table(df_pred), unsafe_allow_html=True)

# ---------------------------------------------------
# PAGE 3 — Anomalies
# ---------------------------------------------------
elif page == "Anomalies":
    st.header("🚨 Anomaly Detection (Top Rare Galaxies)")
    st.markdown("These galaxies appear unusual based on Isolation Forest, LOF, and PCA error.")
    if os.path.exists(IMG_ANOM):
        st.image(IMG_ANOM, caption="Anomaly Detection Map", use_container_width=True)
    else:
        st.error(f"❌ Anomaly image not found at: {IMG_ANOM}")
    unsup_df["score"] = unsup_df["iso_forest_anomaly"] + unsup_df["lof_anomaly"] + unsup_df["pca_anomaly"]
    top10 = unsup_df.sort_values("score", ascending=False).head(10)
    cols = st.columns(5)
    for i, (_, g) in enumerate(top10.iterrows()):
        img = get_img(int(g["galaxy_id"]))
        with cols[i%5]:
            st.image(img, caption=f"ID {g['galaxy_id']}", use_container_width=True)

# ---------------------------------------------------
# PAGE 4 — Clusters
# ---------------------------------------------------
elif page == "Clusters":
    st.header("🌐 Cluster Viewer (K-Means / DBSCAN)")
    st.markdown("Browse galaxies grouped by your clustering algorithms.")
    cluster_type = st.selectbox("Cluster Method", ["kmeans_k3","dbscan_cluster"])
    cluster_counts = unsup_df[cluster_type].value_counts()
    valid_clusters = [c for c in cluster_counts[cluster_counts>5].index if c!=-1]
    cid = st.selectbox("Cluster ID", sorted(valid_clusters))
    group = unsup_df[unsup_df[cluster_type]==cid].head(25)
    cols = st.columns(5)
    for i, (_, g) in enumerate(group.iterrows()):
        img = get_img(int(g["galaxy_id"]))
        with cols[i%5]:
            st.image(img, caption=f"ID {g['galaxy_id']}", use_container_width=True)

# ---------------------------------------------------
# PAGE 5 — t-SNE Map
# ---------------------------------------------------
elif page == "t-SNE Map":
    st.header("🌀 t-SNE Embedding Map")
    st.markdown("Static 2D visualization of galaxy similarity using t-SNE.")
    if os.path.exists(IMG_TSNE):
        st.image(IMG_TSNE, caption="t-SNE Galaxy Embedding", use_container_width=True)
    else:
        st.error(f"❌ t-SNE image not found at: {IMG_TSNE}")