import streamlit as st
import utils.session_controler as sc
from dotenv import load_dotenv

# Start without reopening the browser
# streamlit run .\Main_Page.py --server.headless true

st.set_page_config(
    page_title="Main Page",
)

# creating session state variables
sc.initialize("env_path", ".env")


# loading variables
load_dotenv(dotenv_path=sc.get("env_path"), override=True)


st.header("Main Page")

st.error("TODO: add app explanation")

st.write("Session State:")
st.write(st.session_state)
