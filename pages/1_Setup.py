import streamlit as st
import os
from dotenv import load_dotenv
import utils.session_controler as sc
import utils.variable_controller
from utils.setup.neo4j.neo4j_setup import check_neo4j_aura_connection
from utils.setup.chroma.chroma_setup import check_chroma_connection
from utils.setup.mistralai.mistralai_setup import check_mistralai_connection

st.set_page_config(
    page_title="Setup",
)

# initialize in case of refresh on setup page
sc.initialize("env_path", ".\\credientials\\.env")

# list of required variables
# states: 0 = not set, 1 = set but incorrect, 2 = set and correct
initial_required_variables = [
    {"name": "NEO4J_URI", "state": 0},
    {"name": "NEO4J_USERNAME", "state": 0},
    {"name": "NEO4J_PASSWORD", "state": 0},
    {"name": "CHROMADB_PATH", "state": 0},
    {"name": "MISTRALAI_API_KEY", "state": 0},
]
sc.initialize("setup_required_variables", initial_required_variables)
if sc.get("setup_required_variables") is None:
    sc.set("setup_required_variables", initial_required_variables)

def change_variable_state(variable: str, state: int):
    variables = sc.get("setup_required_variables")
    for var in variables:
        if var["name"] == variable:
            var["state"] = state
    sc.set("setup_required_variables", variables)

def env_to_session(env_path: str):
    load_dotenv(dotenv_path=env_path, override=True)
    for variable in sc.get("setup_required_variables"):
        var = variable["name"]
        if os.getenv(var):
            sc.set(var, os.getenv(var))
        else:
            sc.set(var, None)

### PAGE ###

st.header("Setup")


# Environment variables setup
st.divider()
st.subheader("Setup from .env file")

if st.text_input(".env path", value=sc.get("env_path"), key="new_env_path"):
    sc.set("env_path", sc.get("new_env_path"))
    env_to_session(sc.get("env_path"))

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Reload .env", use_container_width=True):
        env_to_session(sc.get("env_path"))
        st.rerun()

with col2:
    if st.button("Check env variables", use_container_width=True):
        required_env_vars = ["CHROMADB_PATH", "MISTRALAI_API_KEY", "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"]
        if not utils.variable_controller.check_env_variables(required_env_vars):
            st.error("Missing .env variables")
        else:
            st.success("All .env variables are set")

with col3:
    if st.button("Open .env file", use_container_width=True):
        os.system(f'notepad.exe {sc.get("env_path")}')


# Neo4j setup
st.divider()
st.subheader("Neo4j setup")

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
    col_uri, col_user, col_pass = st.columns(3)
    with col_uri:
        uri = st.text_input("URI")
    with col_user:
        username = st.text_input("Username")
    with col_pass:
        password = st.text_input("Password", type="password")

if st.button("Save configuration", use_container_width=True):
    if uri and username and password:
        sc.set("NEO4J_URI", uri)
        sc.set("NEO4J_USERNAME", username)
        sc.set("NEO4J_PASSWORD", password)
        
        st.success("Configuration saved")
    else:
        st.error("Missing fields")

# checking connection
def check_neo4j_connection():
    with st.spinner("Establishing connection..."):
        if check_neo4j_aura_connection(sc.get("NEO4J_URI"), sc.get("NEO4J_USERNAME"), sc.get("NEO4J_PASSWORD")):
            st.success("Neo4j Aura connection established")
            change_variable_state("NEO4J_URI", 2)
            change_variable_state("NEO4J_USERNAME", 2)
            change_variable_state("NEO4J_PASSWORD", 2)
        else:
            st.error("Neo4j Aura connection failed")
            change_variable_state("NEO4J_URI", 0)
            change_variable_state("NEO4J_USERNAME", 0)
            change_variable_state("NEO4J_PASSWORD", 0)

if st.button("Check Neo4j connection", use_container_width=True):
    check_neo4j_connection()

# ChromaDB setup
st.divider()
st.subheader("ChromaDB setup")

chroma_path = st.text_input("ChromaDB path", value=sc.get("CHROMADB_PATH") if sc.get("CHROMADB_PATH") else "")
if chroma_path:
    sc.set("CHROMADB_PATH", chroma_path)

# checking connection
def check_chromadb_connection():
    with st.spinner("Establishing connection..."):
        if check_chroma_connection(sc.get("CHROMADB_PATH")):
            st.success("ChromaDB connection established")
            change_variable_state("CHROMADB_PATH", 2)
        else:
            st.error("ChromaDB connection failed")
            change_variable_state("CHROMADB_PATH", 0)

if st.button("Check ChromaDB connection", use_container_width=True) and sc.get("CHROMADB_PATH"):
    check_chromadb_connection()


# MistralAI setup
st.divider()
st.subheader("MistralAI API Key setup")
mistral_key = st.text_input("MistralAI API Key", value=sc.get("MISTRALAI_API_KEY") if sc.get("MISTRALAI_API_KEY") else "", type="password")
if mistral_key:
    sc.set("MISTRALAI_API_KEY", mistral_key)

# checking connection
def check_mistralai_api_key_connection():
    with st.spinner("Establishing connection..."):
        if check_mistralai_connection(sc.get("MISTRALAI_API_KEY")):
            st.success("MisaltralAI key is valid")
            change_variable_state("MISTRALAI_API_KEY", 2)
        else:
            st.error("MistralAI key is invalid")
            change_variable_state("MISTRALAI_API_KEY", 0)

if st.button("Check MistralAI connection", use_container_width=True) and sc.get("MISTRALAI_API_KEY"):
    check_mistralai_api_key_connection()


### SIDEBAR ###




with st.sidebar:

    # shortcut to check all connections
    if st.button("Check all connections", use_container_width=True):
        if sc.get("NEO4J_URI") and sc.get("NEO4J_USERNAME") and sc.get("NEO4J_PASSWORD"):
            check_neo4j_connection()
        else:
            st.warning("Neo4j variables are not set")
        
        if sc.get("CHROMADB_PATH"):
            check_chromadb_connection()
        else:
            st.warning("ChromaDB path is not set")
        
        if sc.get("MISTRALAI_API_KEY"):
            check_mistralai_api_key_connection()
        else:
            st.warning("MistralAI API Key is not set")

    # provide info for user about required variables
    st.write("Required variables status")
    for variable in sc.get("setup_required_variables"):
        variable_name = variable["name"]
        if sc.check_exists(variable_name) and not sc.check_empty(variable_name) and variable["state"] != 2:
            change_variable_state(variable_name, 1)
        if variable["state"] == 0:
            st.error(f"{variable['name']}")
        elif variable["state"] == 1:
            st.warning(f"{variable['name']}")
        elif variable["state"] == 2:
            st.success(f"{variable['name']}")