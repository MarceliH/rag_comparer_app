import streamlit as st
from mistralai.models.chatcompletionresponse import ChatCompletionResponse
from neo4j_graphrag.generation.types import RagResultModel


class RagReport:
    def __init__(
        self,
        query=None,
        vector_retriever=None,
        graph_retriever=None,
        model_name=None,
        temperature=None,
        max_tokens=None,
        user_template=None,
        system_template=None,
        vector_response=None,
        vector_generation_time=None,
        graph_response=None,
        graph_generation_time=None,
    ):
        self.data = {
            "query": query if query is not None else "",
            "retrieval": {
                "vector_retriever": vector_retriever if vector_retriever is not None else {
                    "context": "",
                    "documents": [],
                    "document_names": [],
                    "distances": [],
                    "retrieval_time": 0.0,
                    "top_k": 0,
                },
                "graph_retriever": graph_retriever if graph_retriever is not None else {
                    "context": "",
                    "graph_data": [],
                    "generated_cypher": "",
                    "expanded": False,
                    "expansion_success": False,
                    "generation_count": 0,
                    "cypher_query": None,
                    "retrieval_time": 0.0,
                    "top_k": 0,
                },
            },
            "generation": {
                "model_name": model_name if model_name is not None else "",
                "temperature": temperature if temperature is not None else 0.0,
                "max_tokens": max_tokens if max_tokens is not None else 0,
                "prompt": {
                    "user_template": user_template if user_template is not None else "",
                    "system_template": system_template if system_template is not None else "",
                },
                "vector_response": vector_response if vector_response is not None else "",
                "vector_generation_time": vector_generation_time if vector_generation_time is not None else 0.0,
                "graph_response": graph_response if graph_response is not None else "",
                "graph_generation_time": graph_generation_time if graph_generation_time is not None else 0.0,
            }
        }

    def fill(self, **kwargs):
        # Query
        if "query" in kwargs and kwargs["query"] is not None:
            self.data["query"] = kwargs["query"]

        # Vector retriever
        if "vector_retriever" in kwargs and kwargs["vector_retriever"] is not None:
            self.data["retrieval"]["vector_retriever"].update(kwargs["vector_retriever"])
        else:
            for key in self.data["retrieval"]["vector_retriever"]:
                if key in kwargs and kwargs[key] is not None:
                    self.data["retrieval"]["vector_retriever"][key] = kwargs[key]

        # Graph retriever
        if "graph_retriever" in kwargs and kwargs["graph_retriever"] is not None:
            self.data["retrieval"]["graph_retriever"].update(kwargs["graph_retriever"])
        else:
            for key in self.data["retrieval"]["graph_retriever"]:
                if key in kwargs and kwargs[key] is not None:
                    self.data["retrieval"]["graph_retriever"][key] = kwargs[key]

        # Generation
        for key in self.data["generation"]:
            if key in kwargs and kwargs[key] is not None:
                if key == "prompt":
                    self.data["generation"]["prompt"].update(kwargs[key])
                else:
                    self.data["generation"][key] = kwargs[key]

    def to_json(self):
        return self.data

    def streamlit_display(self):
        st.subheader("Query")
        st.write(self.data["query"])

        vret = self.data["retrieval"]["vector_retriever"]
        gret = self.data["retrieval"]["graph_retriever"]
        gen = self.data["generation"]

        col_vr, col_gr = st.columns(2)

        with col_vr:
            st.markdown(f"**Vector Response (Time: {gen['vector_generation_time']:.2f} s)**")
            st.write(gen["vector_response"])

        with col_gr:
            st.markdown(f"**Graph Response (Time: {gen['graph_generation_time']:.2f} s)**")
            st.write(gen["graph_response"])

        col_v, col_g = st.columns(2)

        with col_v:
            st.markdown("### Vector Retrieval")
            st.write(f"Retrieval Time: {vret['retrieval_time']:.2f} s")
            st.write(f"Top-k: {vret['top_k']}")
            if vret.get("context"):
                st.write("Context:")
                st.write(vret["context"])
            for i in range(len(vret["document_names"])):
                st.markdown(f"**Doc {i+1}:** {vret['document_names'][i]}")
                st.markdown(f"*Score:* {vret['distances'][i]:.4f}")
                st.write(vret["documents"][i])
            st.markdown(f"**Vector Response (Time: {gen['vector_generation_time']:.2f} s)**")
            st.write(gen["vector_response"])

        with col_g:
            st.markdown("### Graph Retrieval")
            st.write(f"Retrieval Time: {gret['retrieval_time']:.2f} s")
            st.write(f"Top-k: {gret['top_k']}")
            if gret.get("context"):
                st.write("Context:")
                st.write(gret["context"])
            st.write(f"Cypher: {gret['generated_cypher']}")
            st.write(f"Expanded: {gret['expanded']}, Expansion Success: {gret['expansion_success']}")
            st.write(f"Generation Count: {gret['generation_count']}")
            if gret["cypher_query"]:
                st.write("Cypher Query:")
                st.code(gret["cypher_query"])
            if gret["graph_data"]:
                st.write("Graph Data:")
                st.write(gret["graph_data"])

        st.subheader("Generation Settings")
        st.write(f"Model: {gen['model_name']}")
        st.write(f"Temperature: {gen['temperature']}")
        st.write(f"Max Tokens: {gen['max_tokens']}")
        st.write("Prompt Templates:")
        st.write(f"User: {gen['prompt']['user_template']}")
        st.write(f"System: {gen['prompt']['system_template']}")
