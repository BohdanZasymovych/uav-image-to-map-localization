import streamlit as st
import logging
import sys
import os

# Add project root to sys.path to allow imports from app and localization
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.app_ui.components.sidebar import Sidebar
from app.app_ui.components.main_panel import MainPanel
from app.app_ui.components.logger_widget import LoggerWidget, StreamlitLogHandler
from app.app_cli.factories import LocalizationComponentFactory
from app.app_cli.renderer import MapOverlayRenderer

class LocalizationStreamlitApp:
    def __init__(self):
        st.set_page_config(
            page_title="UAV Image-to-Map Localization",
            page_icon="✈️",
            layout="wide"
        )
        self.sidebar = Sidebar()
        self.main_panel = MainPanel()
        self.logger_widget = LoggerWidget()
        self._setup_logging()

    def _setup_logging(self):
        # Configure the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Check if handler already exists to avoid duplicate logs
        if not any(isinstance(h, StreamlitLogHandler) for h in root_logger.handlers):
            handler = StreamlitLogHandler()
            formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)
            
        # Ensure sub-loggers are at INFO level as well
        logging.getLogger("localization").setLevel(logging.INFO)
        logging.getLogger("app").setLevel(logging.INFO)

    def run(self):
        st.title("Autonomous UAV Localization")
        st.markdown("""
        This application calculates a UAV's map coordinates by matching its real-time camera feed 
        against preloaded satellite maps.
        """)

        config = self.sidebar.render()
        uav_img, map_img = self.main_panel.render_uploaders()

        if st.button("Run Localization", disabled=uav_img is None or map_img is None):
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
                        position_px_x=float(localization_result.position_px[0]),
                        position_px_y=float(localization_result.position_px[1]),
                        bbox_width=uav_w,
                        bbox_height=uav_h,
                    )
                    
                    self.main_panel.display_results(match_result, localization_result, map_overlay)
                    
                except Exception as e:
                    st.error(f"An error occurred during localization: {e}")
                    logging.exception(e)

        # Always display logs at the bottom
        st.divider()
        self.logger_widget.display()

if __name__ == "__main__":
    app = LocalizationStreamlitApp()
    app.run()
