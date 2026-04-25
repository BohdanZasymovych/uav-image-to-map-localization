import streamlit as st
from typing import Dict, Any

class Sidebar:
    def __init__(self):
        self.model_options = ["Affine", "Similarity"]
        self.extractor_options = ["SIFT"]

    def render(self) -> Dict[str, Any]:
        with st.sidebar:
            st.header("Localization Settings")
            
            selected_model = st.selectbox("Transformation Model", self.model_options, index=0)
            selected_extractor = st.selectbox("Feature Extractor", self.extractor_options, index=0)
            
            st.subheader("RANSAC Parameters")
            epsilon = st.slider("Epsilon (threshold)", 1.0, 10.0, 3.0, 0.5)
            confidence = st.slider("Confidence", 0.9, 0.999, 0.99, 0.005)
            max_iterations = st.number_input("Max Iterations", 100, 10000, 2000, 100)
            
            st.subheader("Extractor Parameters")
            ratio = st.slider("Lowe's Ratio", 0.5, 1.0, 0.75, 0.05)
            
            return {
                "model": selected_model,
                "extractor": selected_extractor,
                "ransac": {
                    "epsilon": epsilon,
                    "confidence": confidence,
                    "max_iterations": max_iterations
                },
                "sift": {
                    "ratio": ratio
                }
            }
