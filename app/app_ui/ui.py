import streamlit as st
import logging
import sys
import os
import json
import cv2
import numpy as np
import shutil
import io
from pathlib import Path
from datetime import datetime

# Add project root to sys.path to allow imports from app and localization
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.app_ui.components.sidebar import Sidebar
from app.app_ui.components.main_panel import MainPanel
from app.app_ui.components.logger_widget import LoggerWidget, StreamlitLogHandler
from app.app_cli.factories import LocalizationComponentFactory
from app.app_cli.renderer import MapOverlayRenderer
from app.app_cli.logging_utils import LoggingConfigurator

# Minimalistic SVG Icons
ICON_UAV = """<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 12px;"><path d="m2 22 7-5 4 5 2-15L2 22z"/><path d="m9 17 11-7"/></svg>"""
ICON_SAVE = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>"""
ICON_DOWNLOAD = """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>"""

class LocalizationStreamlitApp:
    def __init__(self):
        st.set_page_config(
            page_title="UAV Image-to-Map Localization",
            layout="wide"
        )
        self.sidebar = Sidebar()
        self.main_panel = MainPanel()
        self.logger_widget = LoggerWidget()
        self._setup_logging()
        
        if "last_run" not in st.session_state:
            st.session_state.last_run = None

    def _setup_logging(self):
        LoggingConfigurator.configure(
            log_level="INFO",
            extra_handlers=[StreamlitLogHandler()]
        )

    def run(self):
        st.markdown(f"<h1 style='display: flex; align-items: center;'>{ICON_UAV} Autonomous UAV Localization</h1>", unsafe_allow_html=True)
        st.markdown("""
        This application calculates a UAV's map coordinates by matching its real-time camera feed 
        against preloaded satellite maps.
        """)

        config = self.sidebar.render()
        
        run_button_placeholder = st.empty()
        uav_img, map_img = self.main_panel.render_uploaders()

        if run_button_placeholder.button("Run Localization", disabled=uav_img is None or map_img is None, type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                try:
                    factory = LocalizationComponentFactory(config)
                    pipeline = factory.build_pipeline()
                    
                    logging.info("Starting pipeline execution")
                    match_result, localization_result = pipeline.run(uav_img=uav_img, map_img=map_img)
                    logging.info("Pipeline execution finished")
                    # Render map overlay
                    renderer = MapOverlayRenderer()
                    uav_h, uav_w = uav_img.shape[:2]
                    map_overlay, _, _ = renderer.render(
                        map_img=map_img,
                        transform_matrix=localization_result.transform_matrix,
                        uav_width=uav_w,
                        uav_height=uav_h,
                    )

                    
                    st.session_state.last_run = {
                        "match_result": match_result,
                        "localization_result": localization_result,
                        "map_overlay": map_overlay,
                        "config": config,
                        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
                    }
                    
                except Exception as e:
                    st.error(f"An error occurred during localization: {e}")
                    logging.exception(e)

        if st.session_state.last_run:
            lr = st.session_state.last_run
            self.main_panel.display_results(lr["match_result"], lr["localization_result"], lr["map_overlay"])
            
            st.divider()
            st.markdown(f"<h3 style='display: flex; align-items: center;'>{ICON_SAVE} Export Results</h3>", unsafe_allow_html=True)
            
            st.markdown("Download all artifacts including images and the JSON summary in a single ZIP archive.")
            zip_data = self._prepare_zip(lr)
            st.download_button(
                label="Download ZIP Archive",
                data=zip_data,
                file_name=f"localization_run_{lr['timestamp']}.zip",
                mime="application/zip",
                use_container_width=True
            )

        st.divider()
        self.logger_widget.display()

    def _prepare_zip(self, run_data: dict) -> bytes:
        temp_dir = Path("outputs/temp_zip")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        self._write_files_to_disk(temp_dir, run_data)
        
        zip_path = shutil.make_archive("temp_archive", 'zip', temp_dir)
        
        with open(zip_path, "rb") as f:
            zip_data = f.read()
            
        shutil.rmtree(temp_dir)
        os.remove(zip_path)
        
        return zip_data

    def _write_files_to_disk(self, path: Path, run_data: dict):
        lr = run_data["localization_result"]
        mr = run_data["match_result"]
        cfg = run_data["config"]
        
        cv2.imwrite(str(path / "matches_before_ransac.png"), mr.raw_match_image)
        cv2.imwrite(str(path / "matches_after_ransac.png"), mr.match_image)
        cv2.imwrite(str(path / "map_with_estimated_bbox.png"), run_data["map_overlay"])
        
        summary = {
            "timestamp": run_data["timestamp"],
            "config": cfg,
            "result": {
                "position_px": [float(lr.position_px[0]), float(lr.position_px[1])],
                "n_raw_matches": int(lr.n_raw_matches),
                "n_inliers": int(lr.n_inliers),
                "ransac_iterations": int(lr.ransac_iterations),
                "runtime_s": float(lr.runtime_s),
                "transform_matrix": lr.transform_matrix.tolist(),
            }
        }
        
        with (path / "localization_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

if __name__ == "__main__":
    app = LocalizationStreamlitApp()
    app.run()
