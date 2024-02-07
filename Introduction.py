import streamlit as st
import requests
import pandas as pd
st.set_page_config(
    page_title="Jumps Metrics App | SPESS",
    page_icon="random",
    layout="wide",
    initial_sidebar_state="expanded",
    
)

st.title("Welcome to the Jumps Metrics Data Analysis App.")
st.subheader("In cooperation with the School of Physical Education and Sports Science of Athens.")
st.write("#")
st.sidebar.info("Select a page above.")

col1, col2 = st.columns([2.5,1])
with col1:
    st.markdown("""
        **Brief Description:**

        Jump Metrics for the purpose of research of School of Physical Education and Sports Science of Athens. 
        For this purpose a force platform by PLUX Biosignals has been used to test various jumps trials and collects the raw data of athletes.

        The main scope of this research is to collect raw data, transform and processing them into usable data, calculate and measure various indicators and export final results to create useful conclusions depending of the needs.
        For this reason i choosed to use Streamlit App Framework to create a nice and clean web app let the users to navigate very easy into the functionalities. 
        
        At this point, i should mention that we check various jumps, specifically: Counter Jump, Drop Jump, Isometric, Squad Jump.

        **Navigate to the left menu sidebar to choose a page.**

        """
    )
with col2:
    st.write("")

response = requests.get("https://geth.gr")
st.write(response)