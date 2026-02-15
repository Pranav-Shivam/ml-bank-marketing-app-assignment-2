# Enhanced Streamlit App for Bank Marketing Classification
# M.Tech (S1-25_AIMLCZG565) - Machine Learning Assignment 2
# Dark Mode Compatible Version

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go

from model.models import ClassifierModels

# Configuration
TARGET_COLUMN = "deposit"
st.set_page_config(
    page_title="Bank Marketing ML Classifier",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme-aware custom CSS
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Metric cards - theme aware */
    .stMetric {
        background-color: rgba(128, 128, 128, 0.1);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    /* Headers */
    h1 {
        padding-bottom: 20px;
        border-bottom: 3px solid #1f77b4;
    }
    
    h2 {
        margin-top: 30px;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        font-weight: 600;
    }
    
    .stButton>button:hover {
        background-color: #1557a0;
        border: none;
    }
    
    /* Dataframe styling - ensure visibility in dark mode */
    .dataframe {
        border: 1px solid rgba(128, 128, 128, 0.3);
    }
    
    /* Info boxes - theme aware */
    .info-box {
        background-color: rgba(31, 119, 180, 0.1);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    
    /* Expandable sections */
    .streamlit-expanderHeader {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)


def _clean_uploaded_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess uploaded data"""
    cleaned = df.copy()

    if TARGET_COLUMN in cleaned.columns:
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].replace({"yes": 1, "no": 0})

    if "pdays" in cleaned.columns:
        cleaned["previous_contact"] = cleaned["pdays"].apply(lambda x: 0 if x == -1 else 1)
        cleaned["pdays"] = cleaned["pdays"].replace(-1, 0)

    return cleaned


@st.cache_resource
def _load_and_train():
    """Load data and train all models (cached for performance)"""
    with st.spinner("🔄 Training models... This may take a moment on first load."):
        classifier = ClassifierModels()
        setup = classifier.set_up_for_model_training(verbose=False)

        trained_pipelines = {}
        for model_name, pipeline in setup["model_pipelines"].items():
            trained_pipelines[model_name] = classifier.train_model(
                pipeline,
                setup["X_train"],
                setup["y_train"],
            )
    return classifier, setup, trained_pipelines


def _predict_with_scores(classifier, trained_pipeline, x_eval):
    """Make predictions and get probability scores"""
    y_pred, y_score = classifier.predict_model(trained_pipeline, x_eval)
    return y_pred, y_score


def plot_confusion_matrix_enhanced(cm, model_name):
    """Create a dark mode friendly confusion matrix visualization"""
    # Auto-detect theme (approximate)
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted No', 'Predicted Yes'],
        y=['Actual No', 'Actual Yes'],
        colorscale='Blues',
        text=cm,
        texttemplate='<b>%{text}</b>',
        textfont={"size": 20, "color": "white"},
        showscale=True,
        hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=f'Confusion Matrix - {model_name}', font=dict(size=18)),
        xaxis_title='Predicted Label',
        yaxis_title='Actual Label',
        height=450,
        font=dict(size=14),
        template='plotly',  # Uses theme-aware template
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def plot_metrics_comparison(metrics):
    """Create a theme-aware radar chart for metrics comparison"""
    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [
        metrics['Accuracy'],
        metrics['Precision'],
        metrics['Recall'],
        metrics['F1 Score']
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Model Performance',
        line_color='#1f77b4',
        fillcolor='rgba(31, 119, 180, 0.3)',
        hovertemplate='%{theta}: %{r:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=True,
                gridcolor='rgba(128, 128, 128, 0.3)'
            ),
            angularaxis=dict(
                gridcolor='rgba(128, 128, 128, 0.3)'
            )
        ),
        showlegend=False,
        height=450,
        title=dict(text="Performance Metrics Radar", font=dict(size=18)),
        template='plotly',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def plot_roc_curve(y_true, y_score, model_name):
    """Plot theme-aware ROC curve"""
    if y_score is not None:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC curve (AUC = {roc_auc:.3f})',
            line=dict(color='#1f77b4', width=3),
            hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', width=2, dash='dash'),
            hovertemplate='Random: %{x:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text=f'ROC Curve - {model_name}', font=dict(size=18)),
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=450,
            showlegend=True,
            template='plotly',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='rgba(128, 128, 128, 0.3)'),
            yaxis=dict(gridcolor='rgba(128, 128, 128, 0.3)')
        )
        return fig
    return None


def plot_prediction_distribution(y_pred):
    """Create theme-aware prediction distribution pie chart"""
    pred_counts = pd.Series(y_pred).value_counts()
    
    fig = px.pie(
        values=pred_counts.values,
        names=['No Subscription', 'Subscription'],
        color_discrete_sequence=['#ff7f0e', '#2ca02c'],
        hole=0.4
    )
    fig.update_layout(
        height=350,
        template='plotly',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    return fig


def show_model_info(model_name):
    """Display theme-aware information about selected model"""
    model_descriptions = {
        "Logistic Regression": {
            "icon": "📈",
            "desc": "A linear model for binary classification using sigmoid function",
            "pros": "Fast training, interpretable, good baseline",
            "cons": "Assumes linear relationships",
            "use_case": "Quick predictions with feature importance"
        },
        "Decision Tree": {
            "icon": "🌳",
            "desc": "Tree-based model that splits data based on feature values",
            "pros": "Highly interpretable, handles non-linear data",
            "cons": "Prone to overfitting",
            "use_case": "Explainable predictions with clear decision rules"
        },
        "K-Nearest Neighbors": {
            "icon": "🎯",
            "desc": "Instance-based learning using distance metrics",
            "pros": "No training phase, simple concept",
            "cons": "Slow prediction, sensitive to feature scaling",
            "use_case": "Small datasets with well-defined clusters"
        },
        "Naive Bayes": {
            "icon": "🎲",
            "desc": "Probabilistic classifier based on Bayes theorem",
            "pros": "Fast, works well with small data",
            "cons": "Assumes feature independence",
            "use_case": "Text classification, real-time predictions"
        },
        "Random Forest": {
            "icon": "🌲",
            "desc": "Ensemble of decision trees using bagging",
            "pros": "Robust, handles overfitting well, feature importance",
            "cons": "Less interpretable than single tree",
            "use_case": "High accuracy with balanced performance"
        },
        "XGBoost": {
            "icon": "🚀",
            "desc": "Gradient boosting ensemble with regularization",
            "pros": "State-of-art performance, handles missing data",
            "cons": "Longer training time, many hyperparameters",
            "use_case": "Best overall performance for competitions"
        }
    }
    
    info = model_descriptions.get(model_name, {})
    if info:
        with st.container():
            st.markdown(f"""
            <div class="info-box">
                <h3>{info['icon']} {model_name}</h3>
                <p><strong>Description:</strong> {info['desc']}</p>
                <p><strong>✅ Pros:</strong> {info['pros']}</p>
                <p><strong>⚠️ Cons:</strong> {info['cons']}</p>
                <p><strong>💡 Best Use Case:</strong> {info['use_case']}</p>
            </div>
            """, unsafe_allow_html=True)


def main():
    # Header
    st.title("🏦 Bank Marketing Classification App")
    st.markdown("""
    ### 📊 Machine Learning Assignment 2 | M.Tech (S1-25_AIMLCZG565)
    Train multiple classification models and evaluate their performance on bank marketing data.
    Predict whether a client will subscribe to a term deposit based on marketing campaign features.
    """)
    
    # Load and train models
    classifier, setup, trained_pipelines = _load_and_train()
    model_names = list(trained_pipelines.keys())
    
    # Model Selection on Main Page
    st.markdown("---")
    st.header("⚙️ Select Classification Model")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        selected_model = st.selectbox(
            "Choose a model to evaluate",
            model_names,
            help="Select a classification model from the 6 trained models",
            index=5  # Default to XGBoost (best model)
        )
    
    with col2:
        # Show quick stats for selected model
        model_accuracy = {
            "XGBoost": 0.8625,
            "Random Forest": 0.8531,
            "Logistic Regression": 0.8262,
            "K-Nearest Neighbors": 0.8164,
            "Decision Tree": 0.7922,
            "Naive Bayes": 0.7179
        }
        st.metric(
            "Model Accuracy",
            f"{model_accuracy[selected_model]:.2%}",
            help="Accuracy on test set"
        )
    
    # Sidebar - Dataset Info and Rankings
    with st.sidebar:
        st.markdown("### 🏦 ML Classifier")
        st.markdown("---")
        
        st.subheader("📊 Dataset Info")
        st.info(f"""
        **Training:** {setup['X_train'].shape[0]} samples  
        **Testing:** {setup['X_test'].shape[0]} samples  
        **Features:** {setup['X_train'].shape[1]}  
        **Target:** deposit
        """)
        
        st.markdown("---")
        
        # Quick model comparison
        st.subheader("🏆 Model Rankings")
        st.caption("Ranked by accuracy")
        
        model_rankings = {
            "XGBoost": ("🥇", "86.25%"),
            "Random Forest": ("🥈", "85.31%"),
            "Logistic Regression": ("🥉", "82.62%"),
            "K-Nearest Neighbors": ("④", "81.64%"),
            "Decision Tree": ("⑤", "79.22%"),
            "Naive Bayes": ("⑥", "71.79%")
        }
        
        for model, (medal, acc) in model_rankings.items():
            if model == selected_model:
                st.success(f"{medal} **{model}**: {acc}")
            else:
                st.text(f"{medal} {model}: {acc}")
    
    # Show model information
    show_model_info(selected_model)
    
    # All Models Comparison
    st.markdown("---")
    st.header("🔍 All Models Overview")
    
    # Create comparison cards
    models_data = {
        "XGBoost": {"acc": 0.8625, "f1": 0.8582, "icon": "🚀", "rank": 1},
        "Random Forest": {"acc": 0.8531, "f1": 0.8504, "icon": "🌲", "rank": 2},
        "Logistic Regression": {"acc": 0.8262, "f1": 0.8133, "icon": "📈", "rank": 3},
        "K-Nearest Neighbors": {"acc": 0.8164, "f1": 0.8023, "icon": "🎯", "rank": 4},
        "Decision Tree": {"acc": 0.7922, "f1": 0.7774, "icon": "🌳", "rank": 5},
        "Naive Bayes": {"acc": 0.7179, "f1": 0.6576, "icon": "🎲", "rank": 6}
    }
    
    cols = st.columns(3)
    for idx, (model_name, data) in enumerate(models_data.items()):
        with cols[idx % 3]:
            is_selected = (model_name == selected_model)
            border_color = "#1f77b4" if is_selected else "rgba(128, 128, 128, 0.3)"
            bg_color = "rgba(31, 119, 180, 0.1)" if is_selected else "rgba(128, 128, 128, 0.05)"
            
            st.markdown(f"""
            <div style="
                padding: 15px;
                border-radius: 10px;
                border: 2px solid {border_color};
                background-color: {bg_color};
                margin-bottom: 10px;
                text-align: center;
            ">
                <h3>{data['icon']} {model_name}</h3>
                <p style="margin: 5px 0;"><strong>Rank:</strong> #{data['rank']}</p>
                <p style="margin: 5px 0;"><strong>Accuracy:</strong> {data['acc']:.2%}</p>
                <p style="margin: 5px 0;"><strong>F1 Score:</strong> {data['f1']:.4f}</p>
                {'<p style="color: #1f77b4; font-weight: bold; margin-top: 10px;">✓ SELECTED</p>' if is_selected else ''}
            </div>
            """, unsafe_allow_html=True)
    
    st.caption("💡 Click the dropdown above to switch between models")
    
    # File upload section
    st.markdown("---")
    st.header("📁 Data Upload")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Test Dataset (CSV format)",
            type=["csv"],
            help="Upload a CSV file with the same features as training data"
        )
    
    with col2:
        with st.expander("📋 Required Features"):
            feature_list = setup["X_train"].columns.tolist()
            st.text("\n".join(feature_list[:8]))
            if len(feature_list) > 8:
                st.text(f"... and {len(feature_list) - 8} more")
    
    # Data processing
    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        cleaned_df = _clean_uploaded_data(uploaded_df)
        
        st.success("✅ File uploaded successfully!")
        
        # Data preview
        st.subheader("📊 Uploaded Data Preview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", cleaned_df.shape[0])
        col2.metric("Total Columns", cleaned_df.shape[1])
        col3.metric("Memory", f"{cleaned_df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        with st.expander("👁️ View Data Sample", expanded=False):
            st.dataframe(
                cleaned_df.head(10),
                width='stretch',
                height=300
            )
        
        # Feature validation
        expected_features = setup["X_train"].columns.tolist()
        missing_features = [col for col in expected_features if col not in cleaned_df.columns]
        
        if missing_features:
            st.error(f"❌ **Missing Features:** {', '.join(missing_features)}")
            st.stop()
        
        x_eval = cleaned_df[expected_features]
        y_true = cleaned_df[TARGET_COLUMN] if TARGET_COLUMN in cleaned_df.columns else None
        
    else:
        st.info("ℹ️ No CSV uploaded. Using internal test split for evaluation.")
        x_eval = setup["X_test"]
        y_true = setup["y_test"]
    
    # Make predictions
    st.markdown("---")
    st.header("🔮 Model Predictions")
    
    trained_pipeline = trained_pipelines[selected_model]
    
    with st.spinner(f"Running {selected_model} predictions..."):
        y_pred, y_score = _predict_with_scores(classifier, trained_pipeline, x_eval)
    
    # Predictions display
    prediction_df = pd.DataFrame({
        'Prediction': y_pred,
        'Label': ['No' if p == 0 else 'Yes' for p in y_pred]
    })
    
    if y_score is not None:
        prediction_df['Confidence'] = [f"{score:.4f}" for score in y_score]
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📋 Prediction Results")
        with st.expander("View Predictions (First 20)", expanded=True):
            st.dataframe(
                prediction_df.head(20),
                width='stretch',
                height=400
            )
    
    with col2:
        st.subheader("📈 Distribution")
        fig_pie = plot_prediction_distribution(y_pred)
        st.plotly_chart(fig_pie, width='stretch')
    
    # Evaluation metrics (if ground truth available)
    if y_true is None:
        st.warning("⚠️ Target column 'deposit' not found. Evaluation metrics unavailable.")
        st.stop()
    
    # Calculate metrics
    metrics = classifier.evaluate_model(selected_model, y_true, y_pred, y_score)
    
    st.markdown("---")
    st.header("📊 Evaluation Metrics")
    
    # Metrics display
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            "Accuracy",
            f"{metrics['Accuracy']:.4f}",
            help="Overall correctness"
        )
    
    with col2:
        auc_val = metrics['AUC Score']
        st.metric(
            "AUC",
            f"{auc_val:.4f}" if auc_val else "N/A",
            help="Area Under ROC Curve"
        )
    
    with col3:
        st.metric(
            "Precision",
            f"{metrics['Precision']:.4f}",
            help="TP / (TP + FP)"
        )
    
    with col4:
        st.metric(
            "Recall",
            f"{metrics['Recall']:.4f}",
            help="TP / (TP + FN)"
        )
    
    with col5:
        st.metric(
            "F1 Score",
            f"{metrics['F1 Score']:.4f}",
            help="Harmonic mean"
        )
    
    with col6:
        st.metric(
            "MCC",
            f"{metrics['MCC Score']:.4f}",
            help="Matthews Correlation"
        )
    
    # Visualizations
    st.markdown("---")
    st.header("📈 Performance Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Confusion Matrix",
        "🎯 Metrics Radar",
        "📉 ROC Curve",
        "📋 Classification Report"
    ])
    
    with tab1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm = plot_confusion_matrix_enhanced(cm, selected_model)
        st.plotly_chart(fig_cm, width='stretch')
        
        # Add interpretation
        tn, fp, fn, tp = cm.ravel()
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"✅ **True Negatives:** {tn}")
            st.success(f"✅ **True Positives:** {tp}")
        with col2:
            st.error(f"❌ **False Positives:** {fp}")
            st.error(f"❌ **False Negatives:** {fn}")
    
    with tab2:
        st.subheader("Performance Metrics Radar")
        fig_radar = plot_metrics_comparison(metrics)
        st.plotly_chart(fig_radar, width='stretch')
        st.info("📌 All metrics range from 0 to 1. Higher values indicate better performance.")
    
    with tab3:
        st.subheader("ROC Curve")
        fig_roc = plot_roc_curve(y_true, y_score, selected_model)
        if fig_roc:
            st.plotly_chart(fig_roc, width='stretch')
            st.markdown("""
            **Interpretation:**
            - Diagonal = Random classifier (AUC = 0.5)
            - Top-left corner = Perfect classifier (AUC = 1.0)
            - Higher AUC = Better discrimination ability
            """)
        else:
            st.warning("ROC curve not available for this model.")
    
    with tab4:
        st.subheader("Classification Report")
        report = classification_report(y_true, y_pred, digits=4, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        st.dataframe(
            report_df.style.format("{:.4f}")
                         .background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']),
            width='stretch'
        )
        
        with st.expander("📖 Metric Definitions"):
            st.markdown("""
            - **Precision**: Of predicted positives, how many are correct?
            - **Recall**: Of actual positives, how many did we find?
            - **F1-Score**: Balance between precision and recall
            - **Support**: Number of samples per class
            """)
    
    # Download predictions
    st.markdown("---")
    st.header("💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = prediction_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name=f"{selected_model.replace(' ', '_')}_predictions.csv",
            mime="text/csv",
            width='stretch'
        )
    
    with col2:
        metrics_df = pd.DataFrame([metrics])
        metrics_csv = metrics_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Metrics",
            data=metrics_csv,
            file_name=f"{selected_model.replace(' ', '_')}_metrics.csv",
            mime="text/csv",
            width='stretch'
        )
    
    # Footer
    st.markdown("---")
    st.caption("Machine Learning Assignment 2 | M.Tech (S1-25_AIMLCZG565) | BITS Pilani")
    st.caption("Built with Streamlit 🚀")


if __name__ == "__main__":
    main()