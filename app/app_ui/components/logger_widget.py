import streamlit as st
import logging
from typing import List

class StreamlitLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        if "logs" not in st.session_state:
            st.session_state.logs = []

    def emit(self, record):
        log_entry = self.format(record)
        st.session_state.logs.append(log_entry)

class LoggerWidget:
    def __init__(self):
        if "logs" not in st.session_state:
            st.session_state.logs = []
            
    def display(self):
        with st.expander("Logs & Intermediate Results", expanded=False):
            if st.button("Clear Logs"):
                st.session_state.logs = []
                st.rerun()
            
            # Use a container for logs
            log_text = "\n".join(st.session_state.logs)
            st.text_area("Log Output", value=log_text, height=300, disabled=True, label_visibility="collapsed")
