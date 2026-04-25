import streamlit as st
import cv2
import numpy as np
from typing import Optional, Tuple, Any

class MainPanel:
    def render_uploaders(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        col1, col2 = st.columns(2)
        
        with col1:
            uav_file = st.file_uploader("Upload UAV Image", type=["png", "jpg", "jpeg", "tif"])
            uav_img = self._load_image(uav_file)
            if uav_img is not None:
                st.image(uav_img, caption="UAV Image", use_container_width=True, channels="BGR")
                
        with col2:
            map_file = st.file_uploader("Upload Satellite Map", type=["png", "jpg", "jpeg", "tif"])
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
        st.header("Localization Results")
        
        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Inliers", f"{localization_result.n_inliers} / {localization_result.n_raw_matches}")
        m2.metric("Inlier Ratio", f"{localization_result.n_inliers / localization_result.n_raw_matches:.2f}" if localization_result.n_raw_matches > 0 else "0.00")
        m3.metric("Iterations", localization_result.ransac_iterations)
        m4.metric("Runtime", f"{localization_result.runtime_s:.3f} s")
        
        st.subheader("Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["Estimated Location", "Inlier Matches", "Raw Matches"])
        
        with tab1:
            st.image(map_overlay, caption="Satellite Map with Estimated UAV Bounding Box", use_container_width=True, channels="BGR")
            st.write(f"Estimated Pixel Position: **[{localization_result.position_px[0]:.2f}, {localization_result.position_px[1]:.2f}]**")
            
        with tab2:
            st.image(match_result.match_image, caption="Geometrically Consistent Inlier Matches", use_container_width=True, channels="BGR")
            
        with tab3:
            st.image(match_result.raw_match_image, caption="All Initial Matches", use_container_width=True, channels="BGR")
