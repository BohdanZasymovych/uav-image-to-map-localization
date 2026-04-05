# =============================================================================
# app/ui.py
# -----------------------------------------------------------------------------
# OWNER: Person C.
#
# PURPOSE:
#   Streamlit web application. Wires library components together and provides
#   a user interface for single-image localization and quick evaluation runs.
#   Run with: streamlit run app/ui.py
#
# LAYOUT:
#   Sidebar:
#     - Extractor selector  (SIFT / ORB / SURF)
#     - Model selector      (Affine / Projective / Similarity)
#     - RANSAC sliders      (epsilon, confidence, max_iterations)
#     - Lowe ratio slider
#     - "Run pipeline" button
#
#   Main panel:
#     - Two file uploaders  (UAV image, satellite map)
#     - Feature match image (MatchResult.match_image)
#     - Estimated position  (LocalizationResult.position_geo or position_px)
#     - Metric cards        (inlier ratio, RANSAC iterations, runtime)
#
# REGISTRY PATTERN — the only place concrete classes are imported:
#   EXTRACTORS = {
#       "SIFT": SIFTExtractor,
#       "ORB":  ORBExtractor,
#       "SURF": SURFExtractor,
#   }
#   MODELS = {
#       "Affine":      AffineModel,
#       "Projective":  ProjectiveModel,
#       "Similarity":  SimilarityModel,
#   }
#   Instantiate via: EXTRACTORS[selected_name](ratio=lowe_ratio)
#                    MODELS[selected_name]()
#
# NOTE:
#   Zero algorithm logic belongs here. If you find yourself computing a
#   transformation or filtering matches in this file, move that code to
#   the appropriate library module instead.
# =============================================================================

import streamlit as st

# TODO: implement Streamlit UI
