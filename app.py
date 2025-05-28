"""
Streamlit demo app for Biomedical Active Learning
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure page
st.set_page_config(
    page_title="Biomedical Active Learning Demo",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🧬 Biomedical Active Learning Demo")
    st.markdown("**Active Learning outperforms full-data training**: QBC with just 5 initial samples achieves **MCC 0.942** vs full model's **0.9253** on Breast Cancer dataset.")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Home", "Data Exploration", "Active Learning Demo", "Results Comparison", "Interactive Predictions"]
    )
    
    if page == "Home":
        show_home()
    elif page == "Data Exploration":
        show_data_exploration()
    elif page == "Active Learning Demo":
        show_active_learning_demo()
    elif page == "Results Comparison":
        show_results_comparison()
    elif page == "Interactive Predictions":
        show_interactive_predictions()

def show_home():
    st.header("Welcome to Biomedical Active Learning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Key Results")
        
        # Performance table
        data = {
            "Dataset": ["Breast Cancer", "Breast Cancer", "Blood-Brain Barrier", "Blood-Brain Barrier"],
            "Method": ["QBC Full Model", "QBC First5", "RF Full Model", "QBC First5"],
            "MCC": [0.9254, 0.942, 0.6547, 0.6448],
            "F1 Score": [0.9647, 0.9727, 0.8415, 0.8349],
            "ROC AUC": [0.9958, 0.9965, 0.9174, 0.9152],
            "Initial Samples": ["All (455)", "5", "All (2,232)", "5"]
        }
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
    
    with col2:
        st.subheader("🚀 Key Achievements")
        st.markdown("""
        - **Superior Performance**: QBC with 5 samples outperforms full dataset training
        - **Statistical Parity**: Overlapping confidence intervals confirm AL matches full-data performance  
        - **Rapid Convergence**: Models achieve peak performance within 5-10 iterations
        - **Robust Results**: Consistent performance across 10 independent runs
        """)
    
    st.subheader("📊 Project Overview")
    st.markdown("""
    This project demonstrates how active learning can dramatically reduce annotation requirements 
    in biomedical machine learning while maintaining or improving model performance.
    
    **Datasets:**
    1. **Blood-Brain Barrier Penetration**: 2,790 molecules with SMILES representations
    2. **Breast Cancer Wisconsin**: 569 samples with 30 clinical features
    
    **Active Learning Strategies:**
    - **Random Forest (RF)**: Single learner with uncertainty sampling
    - **Query-by-Committee (QBC)**: Ensemble of 5 diverse classifiers using vote entropy
    """)

def show_data_exploration():
    st.header("📊 Data Exploration")
    st.markdown("Explore the datasets used in our active learning experiments.")
    
    # Placeholder for data exploration
    st.info("Data exploration components will be implemented here.")
    
    # Mock data for demonstration
    st.subheader("Dataset Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("BBB Dataset Size", "2,790 molecules")
        st.metric("BBB Features", "600+ (descriptors + embeddings)")
        
    with col2:
        st.metric("Breast Cancer Size", "569 samples") 
        st.metric("Breast Cancer Features", "30 clinical features")

def show_active_learning_demo():
    st.header("🎯 Active Learning Demo")
    st.markdown("Interactive demonstration of active learning strategies.")
    
    # Configuration options
    col1, col2 = st.columns(2)
    
    with col1:
        dataset = st.selectbox("Select Dataset", ["Breast Cancer", "Blood-Brain Barrier"])
        strategy = st.selectbox("AL Strategy", ["Random Forest", "Query-by-Committee"])
        
    with col2:
        initial_samples = st.slider("Initial Samples", 5, 50, 5)
        batch_size = st.slider("Batch Size", 1, 20, 10)
    
    if st.button("Run Active Learning Simulation"):
        st.info("Active learning simulation will be implemented here.")
        
        # Mock learning curve
        iterations = np.arange(1, 21)
        mcc_scores = 0.5 + 0.4 * (1 - np.exp(-iterations/5)) + np.random.normal(0, 0.02, len(iterations))
        
        fig, ax = plt.subplots()
        ax.plot(iterations, mcc_scores, 'b-', label='MCC Score')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('MCC Score')
        ax.set_title(f'{strategy} Learning Curve on {dataset}')
        ax.legend()
        st.pyplot(fig)

def show_results_comparison():
    st.header("📈 Results Comparison")
    st.markdown("Compare performance across different methods and datasets.")
    
    st.info("Detailed results comparison will be implemented here.")

def show_interactive_predictions():
    st.header("🔮 Interactive Predictions")
    st.markdown("Make predictions with trained models.")
    
    st.info("Interactive prediction interface will be implemented here.")

if __name__ == "__main__":
    main()