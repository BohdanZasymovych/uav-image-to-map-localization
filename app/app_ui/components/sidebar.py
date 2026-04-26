import streamlit as st
from typing import Dict, Any

# Minimalistic SVG Icons
ICON_SETTINGS = """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>"""
ICON_REFRESH = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 4px;"><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M3 21v-5h5"/></svg>"""

class Sidebar:
    def __init__(self):
        self.model_options = ["Affine", "Similarity", "Projective"]

    def render(self) -> Dict[str, Any]:
        with st.sidebar:
            st.markdown(f"<h2 style='display: flex; align-items: center;'>{ICON_SETTINGS} Settings</h2>", unsafe_allow_html=True)
            
            # Reset Button
            if st.button("Reset to Defaults", use_container_width=True):
                st.session_state["model_key"] = "Affine"
                st.session_state["epsilon_key"] = 3.0
                st.session_state["confidence_key"] = 0.99
                st.session_state["max_iter_key"] = 2000
                st.session_state["ratio_key"] = 0.75
                st.rerun()

            selected_model = st.selectbox(
                "Transformation Model", 
                self.model_options, 
                index=0,
                key="model_key",
                help="Similarity (4-DOF): Translation, Rotation, and Uniform Scale.\n\nAffine (6-DOF): Adds Shearing and non-uniform scaling.\n\nProjective (8-DOF): Full perspective homography for tilted camera views."
            )
            
            st.subheader("RANSAC Parameters")
            epsilon = st.slider(
                "Epsilon (threshold)", 1.0, 10.0, 3.0, 0.5,
                key="epsilon_key",
                help="Max pixel distance for a match to be considered an 'inlier'. Lower = higher precision; Higher = more robust to noise."
            )
            confidence = st.slider(
                "Confidence", 0.9, 0.999, 0.99, 0.005,
                key="confidence_key",
                help="The target probability for RANSAC to find a valid set of matches. 0.99 is the standard."
            )
            max_iterations = st.number_input(
                "Max Iterations", 100, 10000, 2000, 100,
                key="max_iter_key",
                help="Limits how many times RANSAC will try to find a solution. Increase for difficult matching scenarios."
            )
            
            st.subheader("Extractor Parameters")
            ratio = st.slider(
                "Lowe's Ratio", 0.5, 1.0, 0.75, 0.05,
                key="ratio_key",
                help="Filters matches by uniqueness. Lower values (e.g., 0.7) are stricter and keep only very distinct feature matches."
            )
            
            return {
                "model": selected_model,
                "extractor": "SIFT",
                "ransac": {
                    "epsilon": epsilon,
                    "confidence": confidence,
                    "max_iterations": max_iterations
                },
                "sift": {
                    "ratio": ratio
                }
            }
