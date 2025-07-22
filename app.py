
import streamlit as st
from home_page import home_page
from trends_page import trends_page
from predict_page import predict_page
import time

st.set_page_config(page_title="EDEN AI", page_icon=":earth_africa:", layout="wide")

# Loading page logic
if 'show_loading' not in st.session_state:
    st.session_state['show_loading'] = True

if st.session_state['show_loading']:
    st.markdown("""
        <div style='display: flex; flex-direction: column; align-items: center; justify-content: center; height: 60vh;'>
            <h1 style='color: #00796b; font-size: 2.5rem; margin-bottom: 1rem;'>EDEN AI</h1>
            <h3 style='color: #004d40; font-weight: 400;'>Aquatic Environmental Intelligence</h3>
            <div style='margin-top: 2rem;'>
                <span style='font-size: 1.2rem; color: #333;'>Initializing application, please wait...</span>
            </div>
            <div style='margin-top: 2rem;'>
                <div class='loader'></div>
            </div>
        </div>
        <style>
        .loader {
          border: 8px solid #e0f2f7;
          border-top: 8px solid #00796b;
          border-radius: 50%;
          width: 60px;
          height: 60px;
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        </style>
    """, unsafe_allow_html=True)
    time.sleep(3)
    st.session_state['show_loading'] = False
    st.rerun()
else:
    page = st.sidebar.radio(
        "Navigate to:",
        ("Home", "Trends Analysis", "Predict Future Events"),
        index=0
    )

    if page == "Home":
        home_page()
    elif page == "Trends Analysis":
        trends_page()
    elif page == "Predict Future Events":
        predict_page()
