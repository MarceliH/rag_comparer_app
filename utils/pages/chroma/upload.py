import streamlit as st
import utils.session_controler as sc
import os
import chromadb
from datetime import datetime
from utils.file_readers import pdf_to_text
from utils.text_splitters import recursive_text_splitter



def add_new_collection(chroma_client):
    with st.expander("New chroma collection", expanded=True):
        new_collection_name = st.text_input("Name")

        collection_description = st.text_input(
            "Description", 
            value="none"
        )

        c1, c2 = st.columns([1,2])
        if new_collection_name and collection_description and c1.button("Create collection"):
            chroma_client.create_collection(
                name=new_collection_name,
                metadata={
                    "description": collection_description,
                    "created": str(datetime.now()),
                } 
            )

            c2.write("Collection created")
            st.rerun()

def chroma_setup(new_collection_option:bool = True):
    chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMADB_PATH") or "C:\\chroma")

    # selecting or adding collection
    collection_options = list(chroma_client.list_collections())
    if new_collection_option:
        collection_options += ["[+ add new]"]

    selected_collection = st.selectbox(
        "Select collection",
        options=collection_options,
        key="selected_collection"
    )
    if selected_collection:
        if selected_collection == "[+ add new]":
            # adding new collection to chroma (ends with rerun)
            add_new_collection(chroma_client)
        else:
            chroma_collection = chroma_client.get_collection(selected_collection)
            return chroma_collection


def chroma_upload(chroma_collection, uploaded_files): 
    # setup text splitter   
    with st.expander("Text splitter setup", expanded=True):
        c1,c2 = st.columns([1,1])
        splitter_chunk_size = c1.number_input(
            "Chunk size",
            min_value=50,
            value=200,
            step=1
        )

        splitter_overlap_percentage = c2.slider(
            "Overlap percent",
            min_value=0,
            max_value=100,
            step=1,
            value=15,
        )
        splitter_chunk_overlap = round(splitter_chunk_size * (splitter_overlap_percentage/100))

    # adding to collection
    if st.button("Add to collection"):

        table_file_names = []
        table_chunk_counts = []

        # splitting documents
        for uploaded_file in uploaded_files:

            file_text = pdf_to_text(uploaded_file)

            splitted_file_text = recursive_text_splitter(
                text=file_text,
                chunk_size=splitter_chunk_size,
                chunk_overlap=splitter_chunk_overlap
            )

            table_file_names.append(uploaded_file.name)
            table_chunk_counts.append(len(splitted_file_text))

            with st.spinner(f"Adding '{uploaded_file.name}' to collection..."):
                ids = []
                tmstp = int(datetime.now().timestamp())
                metadatas = []
                for i, value in enumerate(splitted_file_text):
                    ids.append(f"{i}_{tmstp}")
                    metadatas.append({
                        "source": uploaded_file.name,
                        "chunk": i,
                        "chunk_size": splitter_chunk_size,
                        "chunk_overlap": splitter_chunk_overlap
                    })

                chroma_collection.add(
                    ids=ids,
                    metadatas=metadatas,
                    documents=splitted_file_text
                )

        st.table({
            "Files": table_file_names,
            "Number of chunks": table_chunk_counts
        })


def chroma_query(chroma_collection):
    with st.expander("Query test", expanded=False):
        c1, c2 = st.columns([4,1])
        chroma_query = c1.text_input("Query", label_visibility="collapsed")
        ask = c2.button("Ask")
        if chroma_query and ask:
            response = chroma_collection.query(
                query_texts=[chroma_query],
                n_results=3
            )
            st.write(response)


def chroma_delete_collection(chroma_collection):
    if st.button("Delete collection"):
        st.warning(f'Are you sure you want to delete collection "{chroma_collection.name}"')
        if st.button("Yes"):
            with st.spinner("Deleting..."):
                chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMADB_PATH") or "C:\\chroma")
                chroma_client.delete_collection(chroma_collection.name)
            
            st.rerun()
