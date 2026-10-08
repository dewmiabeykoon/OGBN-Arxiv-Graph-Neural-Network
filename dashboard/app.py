import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

# ----------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME STYLING
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="OGBN-Arxiv Graph Intelligence Dashboard",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Theme Colors
PRIMARY_COLOR = "#3B82F6"
SECONDARY_COLOR = "#6366F1"
SUCCESS_COLOR = "#10B981"
DANGER_COLOR = "#EF4444"

st.markdown(
    f"""
    <style>
    /* Global Layout Tweaks */
    .block-container {{
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }}
    
    /* Typography Hierarchy */
    h1 {{
        color: #F3F4F6;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem !important;
    }}
    h2 {{
        color: #E5E7EB;
        font-weight: 700 !important;
        font-size: 1.55rem !important;
        border-bottom: 2px solid rgba(59, 130, 246, 0.3);
        padding-bottom: 8px;
        margin-top: 2.2rem !important;
        margin-bottom: 1.2rem !important;
    }}
    h3 {{
        color: #D1D5DB;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
        margin-top: 1.5rem !important;
    }}

    /* Glassmorphism Metric Cards */
    div[data-testid="stMetric"] {{
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.2rem 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-3px);
        border-color: {PRIMARY_COLOR};
    }}
    
    /* Metric Value Styling */
    div[data-testid="stMetricValue"] > div {{
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: {PRIMARY_COLOR} !important;
    }}

    /* Consistent Image Containers */
    div[data-testid="stImage"] img {{
        width: 100% !important;
        height: auto !important;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    }}
    div[data-testid="stCaptionContainer"] {{
        text-align: center;
        font-size: 0.88rem;
        color: #9CA3AF;
        margin-top: 8px;
    }}

    /* Custom Status Badges */
    .badge {{
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.92rem;
        text-align: center;
        width: 100%;
    }}
    .badge-success {{
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }}
    .badge-error {{
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }}
    
    /* Sidebar Section Title */
    .sidebar-title {{
        font-weight: 700;
        font-size: 1.1rem;
        color: #F3F4F6;
        margin-bottom: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# 2. HELPERS & LOADERS
# ----------------------------------------------------------------------
def show_figure(filename, caption, narrow=False):
    """Consistent Figure Loader"""
    path = BASE_DIR / filename
    if not path.exists():
        st.warning(f"⚠️ Image `{filename}` is not found in directory.")
        return
    if narrow:
        _, mid, _ = st.columns([1, 4, 1])
        with mid:
            st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.image(str(path), caption=caption, use_container_width=True)


def show_pair(left, right):
    """Side-by-side figures with equal widths"""
    c1, c2 = st.columns(2, gap="medium")
    for col, (filename, caption) in zip((c1, c2), (left, right)):
        with col:
            show_figure(filename, caption)


@st.cache_data
def load_graph_stats():
    with open(BASE_DIR / "graph_stats.json", "r") as f:
        return json.load(f)


@st.cache_data
def load_model_results():
    return pd.read_csv(BASE_DIR / "model_evaluation_results.csv", index_col=0)


@st.cache_data
def load_predictions():
    return pd.read_csv(BASE_DIR / "predictions.csv")


@st.cache_data
def load_csv(name, **kwargs):
    path = BASE_DIR / name
    return pd.read_csv(path, **kwargs) if path.exists() else None


stats = load_graph_stats()
model_results = load_model_results()
predictions = load_predictions()

# ----------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🕸️ Dashboard Navigation")
    st.markdown("Select a topic to navigate directly to that section:")
    
    nav_choice = st.radio(
        "Jump to Section:",
        [
            "1. Graph Statistics",
            "2. Dataset Exploration",
            "3. Model Performance",
            "4. Hyperparameter Optimization",
            "5. Node Classification Results",
            "6. Learned Node Embeddings",
            "7. Sample Citation Subgraph",
            "8. Explainability",
        ],
        index=0,
    )
    
    st.divider()

# ----------------------------------------------------------------------
# MAIN TITLE & HEADER
# ----------------------------------------------------------------------
st.title("OGBN-Arxiv Graph Intelligence Dashboard")

st.info(
    "💡 **Overview:** This dashboard provides end-to-end insights into graph statistics, "
    "Deep Graph Learning (GCN vs. GraphSAGE) performance metrics, node classification results, "
    "and embedding visualizations for the **OGBN-Arxiv citation network**."
)

# ----------------------------------------------------------------------
# 1. GRAPH STATISTICS
# ----------------------------------------------------------------------
if nav_choice == "1. Graph Statistics" or nav_choice is not None:
    st.header("1. Graph Statistics")

    r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
    r1_c1.metric("Nodes", f"{stats['nodes']:,}")
    r1_c2.metric("Directed Edges", f"{stats['directed_edges']:,}")
    r1_c3.metric("Node Features", stats["features"])
    r1_c4.metric("Classes", stats["classes"])

    st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)

    r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
    r2_c1.metric("Message-Passing Edges", f"{stats['message_passing_edges']:,}")
    r2_c2.metric("Training Nodes", f"{stats['training_nodes']:,}")
    r2_c3.metric("Validation Nodes", f"{stats['validation_nodes']:,}")
    r2_c4.metric("Test Nodes", f"{stats['test_nodes']:,}")

    st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
    st.metric("Directed Graph Density", f"{stats['density']:.8f}")
    st.caption(
        "📌 *Note: The extremely low graph density indicates that OGBN-Arxiv is a highly sparse citation network.*"
    )

# ----------------------------------------------------------------------
# 2. DATASET EXPLORATION
# ----------------------------------------------------------------------
if nav_choice == "2. Dataset Exploration" or nav_choice is not None:
    st.divider()
    st.header("2. Dataset Exploration")

    show_figure("class_distribution.png", "Class distribution across the 40 subject areas")
    show_pair(
        ("in_degree_distribution.png", "In-degree distribution"),
        ("out_degree_distribution.png", "Out-degree distribution"),
    )
    show_figure("node_feature_distribution.png", "Node feature distribution", narrow=True)

# ----------------------------------------------------------------------
# 3. MODEL PERFORMANCE
# ----------------------------------------------------------------------
if nav_choice == "3. Model Performance" or nav_choice is not None:
    st.divider()
    st.header("3. Model Performance")
    st.markdown(
        "Comparison of **GCN** and **GraphSAGE** performance metrics across validation and test subsets."
    )

    st.dataframe(model_results.round(4), use_container_width=True)

    # Plotly Chart for Graph Consistency
    chart_cols = [c for c in ["Accuracy", "F1_Weighted", "F1_Macro"] if c in model_results.columns]
    if chart_cols:
        st.subheader("Performance Metric Comparison")
        
        # Reshape data for Plotly consistency
        df_melted = model_results[chart_cols].reset_index().melt(
            id_vars=model_results.index.name or "index",
            value_vars=chart_cols,
            var_name="Metric",
            value_name="Score"
        )
        
        fig = px.bar(
            df_melted,
            x=df_melted.columns[0],
            y="Score",
            color="Metric",
            barmode="group",
            color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR],
            template="plotly_dark",
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    show_figure("model_performance_comparison.png", "Model performance comparison chart")

    st.subheader("Training Curves")
    show_pair(
        ("gcn_training_loss.png", "GCN Training Loss"),
        ("graphsage_training_loss.png", "GraphSAGE Training Loss"),
    )
    show_pair(
        ("gcn_accuracy_curve.png", "GCN Accuracy Curve"),
        ("graphsage_accuracy_curve.png", "GraphSAGE Accuracy Curve"),
    )

    st.subheader("Normalized Confusion Matrices")
    show_pair(
        ("gcn_normalized_confusion_matrix.png", "GCN Confusion Matrix"),
        ("graphsage_normalized_confusion_matrix.png", "GraphSAGE Confusion Matrix"),
    )

    st.subheader("Per-Class Reports")
    r1, r2 = st.columns(2, gap="medium")
    for col, name, label in (
        (r1, "gcn_classification_report.csv", "GCN"),
        (r2, "graphsage_classification_report.csv", "GraphSAGE"),
    ):
        with col:
            with st.expander(f"📋 {label} Classification Report"):
                df = load_csv(name, index_col=0)
                if df is not None:
                    st.dataframe(df.round(4), use_container_width=True, height=400)

# ----------------------------------------------------------------------
# 4. HYPERPARAMETER OPTIMIZATION
# ----------------------------------------------------------------------
if nav_choice == "4. Hyperparameter Optimization" or nav_choice is not None:
    st.divider()
    st.header("4. Hyperparameter Optimization")

    hp = load_csv("hyperparameter_tuning_results.csv")
    if hp is not None:
        st.dataframe(hp.round(4), use_container_width=True, hide_index=True)
    show_figure("hyperparameter_optimization_comparison.png", "Hyperparameter optimization comparison")

# ----------------------------------------------------------------------
# 5. NODE CLASSIFICATION RESULTS
# ----------------------------------------------------------------------
if nav_choice == "5. Node Classification Results" or nav_choice is not None:
    st.divider()
    st.header("5. Node Classification Results")
    st.markdown(
        "Select a Node ID to inspect ground-truth labels and model prediction correctness."
    )

    node_id = st.number_input(
        "Enter Node ID", min_value=0, max_value=len(predictions) - 1, value=0, step=1
    )
    selected_row = predictions[predictions["Node_ID"] == node_id]

    if not selected_row.empty:
        row = selected_row.iloc[0]
        cols = st.columns(4)
        cols[0].metric("True Class", int(row["True_Label"]))
        cols[1].metric("GCN Prediction", int(row["GCN_Prediction"]))
        cols[2].metric("GraphSAGE Prediction", int(row["GraphSAGE_Prediction"]))
        cols[3].metric("Dataset Split", row["Split"])

        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        ok1, ok2 = st.columns(2)
        
        gcn_correct = bool(row['GCN_Correct'])
        sage_correct = bool(row['GraphSAGE_Correct'])
        
        with ok1:
            cls = "badge-success" if gcn_correct else "badge-error"
            txt = "GCN Prediction: ✔ Correct" if gcn_correct else "GCN Prediction: ✖ Incorrect"
            st.markdown(f"<div class='badge {cls}'>{txt}</div>", unsafe_allow_html=True)
            
        with ok2:
            cls = "badge-success" if sage_correct else "badge-error"
            txt = "GraphSAGE Prediction: ✔ Correct" if sage_correct else "GraphSAGE Prediction: ✖ Incorrect"
            st.markdown(f"<div class='badge {cls}'>{txt}</div>", unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        st.dataframe(selected_row, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ Selected Node ID was not found.")

    st.subheader("Browse Node Predictions")
    selected_split = st.selectbox(
        "Filter by Dataset Split", ["All", "Training", "Validation", "Test"]
    )
    filtered = (
        predictions if selected_split == "All"
        else predictions[predictions["Split"] == selected_split]
    )
    st.dataframe(filtered.head(100), use_container_width=True, hide_index=True)
    st.caption("Displaying first 100 node records.")

# ----------------------------------------------------------------------
# 6. LEARNED NODE EMBEDDINGS
# ----------------------------------------------------------------------
if nav_choice == "6. Learned Node Embeddings" or nav_choice is not None:
    st.divider()
    st.header("6. Learned Node Embeddings")
    st.markdown(
        "PCA reduction of learned GCN hidden representations into a 2D projection."
    )
    show_figure("pca_embeddings.png", "PCA visualization of learned GCN node embeddings")

# ----------------------------------------------------------------------
# 7. SAMPLE CITATION SUBGRAPH
# ----------------------------------------------------------------------
if nav_choice == "7. Sample Citation Subgraph" or nav_choice is not None:
    st.divider()
    st.header("7. Sample Citation Subgraph")
    show_figure("sample_subgraph.png", "Sample topology of the directed OGBN-Arxiv citation network")

# ----------------------------------------------------------------------
# 8. EXPLAINABILITY
# ----------------------------------------------------------------------
if nav_choice == "8. Explainability" or nav_choice is not None:
    st.divider()
    st.header("8. Explainability")

    exp = load_csv("explainability_summary.csv")
    if exp is not None:
        exp["Value"] = exp["Value"].astype(float).round(4)
        st.dataframe(exp, use_container_width=True, hide_index=True)

    show_pair(
        ("edge_ablation_confidence.png", "Effect of edge removal on GCN confidence"),
        ("neighborhood_influence.png", "Neighborhood influence on predictions"),
    )

# ----------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------
st.divider()
st.markdown(
    "<div style='text-align: center; color: #6B7280; font-size: 0.85rem; padding-bottom: 1rem;'>"
    "Graph Intelligence Dashboard • </b> "
    "using OGBN-Arxiv Network."
    "</div>",
    unsafe_allow_html=True,
)