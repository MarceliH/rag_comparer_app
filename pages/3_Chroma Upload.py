import streamlit as st
import utils.session_controler as sc
from utils.pages.chroma.upload import chroma_upload, chroma_setup, chroma_query, chroma_delete_collection

st.set_page_config(
    page_title="Chroma Ingest Page",
    layout="wide"
)


# PAGE
uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf"],
        accept_multiple_files=True,
    )

left, right = st.columns([1, 1])


chroma_collection = chroma_setup()
if sc.get("selected_collection") != "[+ add new]":
    if uploaded_files:
        chroma_upload(chroma_collection, uploaded_files)
    
    else:
        chroma_query(chroma_collection)
        chroma_delete_collection(chroma_collection)


    
    
