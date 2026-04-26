import streamlit as st
import cv2
import numpy as np
from typing import Optional, Tuple, Any

# Minimalistic SVG Icons
ICON_UPLOAD = """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>"""
ICON_TARGET = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>"""
ICON_PIN = """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>"""

class MainPanel:
    def render_uploaders(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with st.expander("Image Upload", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"<div style='display: flex; align-items: center; margin-bottom: 8px;'>{ICON_UPLOAD} <b>UAV Image</b></div>", unsafe_allow_html=True)
                uav_file = st.file_uploader("Upload UAV Image", type=["png", "jpg", "jpeg", "tif"], label_visibility="collapsed")
                uav_img = self._load_image(uav_file)
                if uav_img is not None:
                    st.image(uav_img, caption="UAV Image", use_container_width=True, channels="BGR")
                    
            with col2:
                st.markdown(f"<div style='display: flex; align-items: center; margin-bottom: 8px;'>{ICON_UPLOAD} <b>Satellite Map</b></div>", unsafe_allow_html=True)
                map_file = st.file_uploader("Upload Satellite Map", type=["png", "jpg", "jpeg", "tif"], label_visibility="collapsed")
                map_img = self._load_image(map_file)
                if map_img is not None:
                    st.image(map_img, caption="Satellite Map", use_container_width=True, channels="BGR")
                    
        return uav_img, map_img

    def _load_image(self, uploaded_file) -> Optional[np.ndarray]:
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            return img
        return None

    def display_results(self, match_result: Any, localization_result: Any, map_overlay: np.ndarray):
        st.markdown(f"<h2 style='display: flex; align-items: center;'>{ICON_TARGET} Localization Results</h2>", unsafe_allow_html=True)
        
        # Main result: Estimated Position
        st.markdown(f"<h3 style='display: flex; align-items: center;'>{ICON_PIN} Estimated Map Position: ({localization_result.position_px[0]:.2f}, {localization_result.position_px[1]:.2f})</h3>", unsafe_allow_html=True)
        
        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Inliers", f"{localization_result.n_inliers} / {localization_result.n_raw_matches}")
        m2.metric("Inlier Ratio", f"{localization_result.n_inliers / localization_result.n_raw_matches:.2f}" if localization_result.n_raw_matches > 0 else "0.00")
        m3.metric("Iterations", localization_result.ransac_iterations)
        m4.metric("Runtime", f"{localization_result.runtime_s:.3f} s")
        
        with st.expander("Visualizations", expanded=True):
            tab1, tab2, tab3 = st.tabs(["Estimated Location", "Inlier Matches", "Raw Matches"])
            
            with tab1:
                st.image(map_overlay, caption="Satellite Map with Estimated UAV Bounding Box", use_container_width=True, channels="BGR")
                
            with tab2:
                st.image(match_result.match_image, caption="Geometrically Consistent Inlier Matches", use_container_width=True, channels="BGR")
                
            with tab3:
                st.image(match_result.raw_match_image, caption="All Initial Matches", use_container_width=True, channels="BGR")
