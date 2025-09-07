import streamlit as st
import utils.session_controler as sc
from utils.raport_parser import RagReport, RagReportSaver
from utils.rag.generation.mistralai_generation import get_available_models, llm_rag_query, rag_system_prompt_template, rag_user_prompt_template
from utils.rag.retrieval.chroma_retrieval import chroma_retriever, get_chroma_collections, instantiate_chroma_client
from utils.rag.retrieval.neo4j_graph_retrieval import instantiate_neo4j_graph, neo4j_text2cypher_retriever, cypher_generation_prompt_template
import pandas as pd
import json
import time


st.set_page_config(
    page_title="Experiment",
    layout="wide"
)

with st.spinner("Connecting..."):
    # Initialize ChromaDB client
    if "chroma_client" not in st.session_state:
        chroma_client = instantiate_chroma_client(chroma_path=sc.gets("CHROMADB_PATH"))
        if chroma_client is None:
            st.error("ChromaDB client could not be instantiated. Please check the CHROMADB_PATH variable in Setup page.")
            st.stop()
        st.session_state.chroma_client = chroma_client
    else:
        chroma_client = st.session_state.chroma_client

    # Initialize Neo4j graph
    if "neo4j_graph" not in st.session_state:
        neo4j_graph = instantiate_neo4j_graph(
            uri=sc.gets("NEO4J_URI"),
            username=sc.gets("NEO4J_USERNAME"),
            password=sc.gets("NEO4J_PASSWORD"),
        )
        if neo4j_graph is None:
            st.error("Neo4j graph could not be instantiated. Please check the Neo4j connection variables in Setup page.")
            st.stop()
        st.session_state.neo4j_graph = neo4j_graph
    else:
        neo4j_graph = st.session_state.neo4j_graph


## PAGE ##
st.title("Experiment page")

left_col, right_col = st.columns([1,1])

#input area for queries
with left_col:
    sc.initialize("queries",[""] * 5)

    st.subheader("Queries")

    # File input
    uploaded_file = st.file_uploader("Upload TXT file with questions (separated by ';')", type=["txt"])
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8")
        queries = [q.strip() for q in content.split(";") if q.strip()]
        sc.set("queries", queries)
        st.success(f"Loaded {len(queries)} questions from TXT.")

    # Manual input
    st.write("Manual input")

    queries = []
    for i, query_val in enumerate(st.session_state.queries):
        col_query, col_remove = st.columns([5, 1])
        with col_query:
            query = st.text_input(f"Query {i+1}", value=query_val, key=f"query_{i}", label_visibility="collapsed")
            st.session_state.queries[i] = query
        with col_remove:
            if st.button("🗑️", key=f"remove_{i}"):
                st.session_state.queries.pop(i)
                st.rerun()
        queries.append(query)

    if st.button("Add query", icon="➕"):
        st.session_state.queries.append("")
        st.rerun()

# settings area
with right_col:

    # settings for vector store
    st.subheader("Vector retireval settings")
    col1, col2 = st.columns(2)

    with col1:
        selected_chroma_collection = st.selectbox(
            "Select Chroma collection",
            options=get_chroma_collections(chroma_client)
        )
    with col2:
        top_k_vector = st.slider(
            "Top-k results",
            min_value=1, max_value=50, value=5, step=1
        )

    # settings for graph database
    st.divider()  
    st.subheader("Graph retrieval settings")
    graph_retriever_type = st.selectbox(
            "Graph retriever type",
            options=["vector", "text2cypher"],
            index=0
        )
    col_graph1, col_graph2 = st.columns(2)

    with col_graph1:
        if graph_retriever_type == "vector":
            top_k_graph = st.number_input(
                "Top-k results",
                min_value=1, max_value=20, value=5, step=1
            )

        if graph_retriever_type == "text2cypher":
            top_k_graph = st.number_input(
                "Top-k results",
                min_value=1, value=50, step=1
            )
            graph_generations = st.slider(
                "Max Cypher generations",
                min_value=1, max_value=5, value=1, step=1
            )
            expand_graph = st.checkbox(
                "Expand graph results", value=True
            )
            with st.expander("Cypher generation prompt"):
                cypher_generation_prompt = st.text_area(
                    "Cypher generation prompt",
                    value=cypher_generation_prompt_template,
                    height=200,
                    label_visibility="collapsed"
                )
                st.write("Use '{schema}' to represent the database schema and '{question}' to represent the user question.")

    with col_graph2:
        traverse_graph_depth = st.slider(
            "Graph traversal depth",
            min_value=1, max_value=5, value=2, step=1
        )

    
    # settings for LLM
    st.divider()  
    st.subheader("LLM settings")
    col_model1, col_model2 = st.columns([3, 1])
    with col_model1:
        model_name = st.text_input(
            "Model name",
            value="mistral-small-2506",
        )
    with col_model2:
        show_models = st.checkbox("Show models", value=False)
    
    col_var1, col_var2 = st.columns([3, 2])
    with col_var1:
        use_temperature = st.checkbox("Use temperature sampling", value=False)
        if use_temperature:
            temperature = st.slider(
                "Temperature",
                min_value=0.0, max_value=2.0, value=0.3, step=0.01
            )
    with col_var2:
        use_max_tokens = st.checkbox("Limit tokens", value=False)
        if use_max_tokens:
            max_tokens = st.number_input(
                "Max tokens",
                min_value=1, max_value=4096, value=512, step=1
            )


    with st.expander("Prompts"):
        col_prompt1, col_prompt2 = st.columns(2)
        with col_prompt1:
            user_prompt = st.text_area(
                "User prompt)",
                value=rag_user_prompt_template,
                height=200
            )
        with col_prompt2:
            system_prompt = st.text_area(
                "System prompt",
                value=rag_system_prompt_template,
                height=200
            )
        st.write("Use '{query}' to represent the user question and '{context}' to represent the retrieved context.")

    

if show_models:
    st.markdown("https://docs.mistral.ai/getting-started/models/models_overview/")

    mistralai_models = get_available_models(sc.gets("MISTRALAI_API_KEY"))
    data = json.loads(mistralai_models)
    df = pd.json_normalize(data["data"])
    st.dataframe(df)
    


# start querying
st.divider()
if st.button("Run experiment", icon="🚀", use_container_width=True):
    for query in queries:
        if not query.strip():
            continue  

        with st.spinner(f"Processing query: {query}"):
            # Graph retrieval
            with st.spinner("Retrieving from graph database..."):
                if graph_retriever_type == "text2cypher":
                    graph_retrieval_result = neo4j_text2cypher_retriever(
                        graph=neo4j_graph,
                        llm_model=model_name,
                        llm_key=sc.gets("MISTRALAI_API_KEY"),
                        query_text=query,
                        top_k=top_k_graph,
                        cypher_generation_prompt=cypher_generation_prompt,
                        max_generations=graph_generations,
                        expand=expand_graph,
                        traversal_depth=traverse_graph_depth,
                    )
                    graph_retriever_type_val = "text2cypher"
                else:
                    from utils.rag.retrieval.neo4j_graph_retrieval import neo4j_vector_retriever
                    graph_retrieval_result = neo4j_vector_retriever(
                        graph=neo4j_graph,
                        query_text=query,
                        top_k=top_k_graph,
                        traversal_depth=traverse_graph_depth,
                    )
                    graph_retriever_type_val = "vector"

            # Vector retrieval
            with st.spinner("Retrieving from vector store..."):
                vector_retrieval_result = chroma_retriever(
                    chroma_client,
                    query,
                    selected_chroma_collection,
                    top_k_vector,
                )

            # Graph generation
            with st.spinner("Generating response from graph context..."):
                graph_generation_start = time.time()
                graph_response_text = llm_rag_query(
                    sc.gets("MISTRALAI_API_KEY"),
                    query,
                    model_name,
                    context=graph_retrieval_result["context"],
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=temperature if use_temperature else None,
                    max_tokens=max_tokens if use_max_tokens else None,
                    use_temperature=use_temperature
                )
                graph_generation_time = time.time() - graph_generation_start

            # Vector generation
            with st.spinner("Generating response from vector store context..."):
                vector_generation_start = time.time()
                vector_response_text = llm_rag_query(
                    sc.gets("MISTRALAI_API_KEY"),
                    query,
                    model_name,
                    context=vector_retrieval_result["documents"],
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=temperature if use_temperature else None,
                    max_tokens=max_tokens if use_max_tokens else None,
                    use_temperature=use_temperature
                )
                vector_generation_time = time.time() - vector_generation_start

            # Fill RagReport
            report = RagReport(
                query=query,
                vector_retriever={
                    "context": vector_retrieval_result.get("context", ""),
                    "documents": vector_retrieval_result.get("documents", []),
                    "document_names": vector_retrieval_result.get("document_names", []),
                    "distances": vector_retrieval_result.get("distances", []),
                    "retrieval_time": vector_retrieval_result.get("retrieval_time", 0.0),
                    "top_k": vector_retrieval_result.get("top_k", top_k_vector),
                },
                graph_retriever={
                    "type": graph_retriever_type_val,
                    "context": graph_retrieval_result.get("context", ""),
                    "graph_data": graph_retrieval_result.get("graph_data", []),
                    "generated_cypher": graph_retrieval_result.get("generated_cypher", ""),
                    "expanded": graph_retrieval_result.get("expanded", False),
                    "expansion_success": graph_retrieval_result.get("expansion_success", False),
                    "generation_count": graph_retrieval_result.get("generation_count", 0),
                    "cypher_query": graph_retrieval_result.get("cypher_query", None),
                    "retrieval_time": graph_retrieval_result.get("retrieval_time", 0.0),
                    "top_k": top_k_graph,
                },
                model_name=model_name,
                temperature=temperature if use_temperature else 0.0,
                max_tokens=max_tokens if use_max_tokens else 0,
                user_template=user_prompt,
                system_template=system_prompt,
                vector_response=vector_response_text,
                vector_generation_time=vector_generation_time,
                graph_response=graph_response_text,
                graph_generation_time=graph_generation_time,
            )

            # Display report
            with st.expander(f"Report for query: {query}"):
                report.streamlit_display()

            # Save report to session state
            sc.initialize("rag_reports", [])
            st.session_state.rag_reports.append(report)
            
    st.rerun()

if "rag_reports" in st.session_state and st.session_state.rag_reports:
    for rep in st.session_state.rag_reports:
        with st.expander(f"Report for query: {rep.data['query']}"):
            rep.streamlit_display()
    
    st.write("Save reports")

    rep1, rep2, rep3 = st.columns([1,1,2])
    with rep1:
        save_path = st.text_input("Enter file path to save reports", placeholder="/path/to/save", label_visibility="collapsed")

    with rep2:
        if st.button("Save to file", icon="💾", use_container_width=True):
            if not st.session_state.rag_reports:
                st.warning("No reports to save.")
            else:
                saver = RagReportSaver(st.session_state.rag_reports)
                if save_path:
                    filepath = saver.save_json(save_path)
                    st.success(f"Reports saved to '{filepath}'.")
                else:
                    st.warning("Please provide a valid file path.")








