# resurces for rag
# https://github.com/langchain-ai/langchain-community/blob/main/libs/community/langchain_community/chains/graph_qa/prompts.py
# https://api.python.langchain.com/en/latest/_modules/langchain/chains/graph_qa/cypher.html#GraphCypherQAChain
# https://neo4j.com/labs/genai-ecosystem/vector-search/
# https://neo4j.com/labs/genai-ecosystem/graphrag-python/
import streamlit as st
import utils.session_controler as sc
import os
import neo4j
from neo4j_graphrag.llm import MistralAILLM
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
import tempfile
import shutil
import asyncio


prompt_template = '''
You are a researcher tasked with extracting information from documents
and structuring it in a property graph to inform further knowledge discovery and Q&A.

Extract the entities (nodes) and specify their type from the following Input text.
Also extract the relationships between these nodes. the relationship direction goes from the start node to the end node.


Return result as JSON using the following format:
{{"nodes": [ {{"id": "0", "label": "the type of entity", "properties": {{"name": "name of entity" }} }}],
  "relationships": [{{"type": "TYPE_OF_RELATIONSHIP", "start_node_id": "0", "end_node_id": "1", "properties": {{"details": "Description of the relationship"}} }}] }}

...

Use only fhe following nodes and relationships:
{schema}

Assign a unique ID (string) to each node, and reuse it to define relationships.
Do respect the source and target node types for relationship and the relationship direction.

Do not return any additional information other than the JSON in it.

Examples:
{examples}

Input text:

{text}
'''


def neo4j_setup():
    connection_options = ["local","Aura instance"]
    selected_connection = st.selectbox("Source", connection_options)

    if selected_connection == connection_options[0]:
        sc.set("neo4j_connection","local")
        with st.expander("Local connection setup", expanded=True):
            uri = st.text_input("URI", "neo4j://localhost")
            username = st.text_input("Username", "neo4j")
            password = st.text_input("Password")

    if selected_connection == connection_options[1]:
        sc.set("neo4j_connection","aura")
        with st.expander("Aura connection setup", expanded=True):
            config = st.file_uploader("Upload Aura credentials file", type=['txt'])
            
            if config is not None:
                content = config.getvalue().decode()                
                for line in content.split('\n'):
                    if 'NEO4J_URI=' in line:
                        uri = line.split('=')[1].strip()
                    elif 'NEO4J_USERNAME=' in line:
                        username = line.split('=')[1].strip()
                    elif 'NEO4J_PASSWORD=' in line:
                        password = line.split('=')[1].strip()
            else:
                uri = st.text_input("URI")
                username = st.text_input("Username")
                password = st.text_input("Password")
    
    # checking connection
    if uri and username and password:
        with st.spinner("Establishing connection..."):
            with neo4j.GraphDatabase.driver(uri, auth=(username, password)) as driver:
                try:
                    driver.verify_connectivity()
                    st.info("Connection established.")

                    os.environ["NEO4J_URI"] = uri
                    os.environ["NEO4J_USERNAME"] = username
                    os.environ["NEO4J_PASSWORD"] = password

                    return True
                except Exception as e:
                    st.error(f"Connection failed: {e}")

                    return False


def knowledge_graph_builder(uploaded_files):
    if not all(key in os.environ for key in ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]):
        st.info("Set up Neo4j connection to upload files")
        return

    with st.expander("Document processing setup", expanded=True):
        llm_name = st.text_input("LLM (from MistralAI)", value="mistral-medium-2505")

        entity_labels_input = st.text_input("Entity labels", placeholder="Separate with ,")
        entity_labels = [label.strip() for label in entity_labels_input.split(",")] if entity_labels_input else []

        relations_input = st.text_input("Relations between entities", placeholder="Separate with ,")
        relations = [relation.strip() for relation in relations_input.split(",")] if relations_input else []   

        c1,c2 = st.columns([1,1])
        splitter_chunk_size = c1.number_input(
            "Chunk size",
            min_value=50,
            value=500,
            step=1,
            key="Chunk size neo4j"
        )
        splitter_overlap_percentage = c2.slider(
            "Overlap percent",
            min_value=0,
            max_value=100,
            step=1,
            value=20,
            key="Overlap percent neo4j"
        )
        splitter_chunk_overlap = round(splitter_chunk_size * (splitter_overlap_percentage/100))
    
    if st.button("Generate knowledge graph"):
        if not all([llm_name, entity_labels, relations, splitter_chunk_size, splitter_chunk_overlap]):
            st.warning("Please fill in all required fields")
        else:
            with st.spinner("Connecting to Neo4j..."):
                neo4j_driver = neo4j.GraphDatabase.driver(
                    os.environ["NEO4J_URI"],
                    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
                )

                mistralai_llm = MistralAILLM(
                    model_name="mistral-small-latest",
                    api_key=os.environ["MISTRALAI_API_KEY"]
                )

                kg_builder_pdf = SimpleKGPipeline(
                    llm=mistralai_llm,
                    driver=neo4j_driver,
                    text_splitter=FixedSizeSplitter(chunk_size=splitter_chunk_size, chunk_overlap=splitter_chunk_overlap),
                    embedder=SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2"),
                    entities=entity_labels,
                    relations=relations,
                    prompt_template=prompt_template,
                    from_pdf=True
                )
                    
            for uploaded_file in uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    temp_dir = None
                    try:
                        temp_dir = tempfile.mkdtemp()
                        path = os.path.join(temp_dir, uploaded_file.name)
                        with open(path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                            
                        pdf_result = asyncio.run(kg_builder_pdf.run_async(file_path=path))
                        st.write(f"Result: {pdf_result}")
                    except Exception as error:
                        st.error(f"Error processing {uploaded_file.name}: {error}")
                        print(f"Error:\n{error}")

                    finally:
                        if temp_dir and os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir)    