import json
import csv
import streamlit as st
from mistralai.models.chatcompletionresponse import ChatCompletionResponse
from neo4j_graphrag.generation.types import RagResultModel
import os
from datetime import datetime
import pandas as pd


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

        col_l1, col_r1 = st.columns(2)

        with col_l1:
            st.subheader(f"Vector Response")
            st.write(gen["vector_response"])
            st.write(f"Generation Time: {gen['vector_generation_time']:.2f} s")


        with col_r1:
            st.subheader(f"Graph Response")
            st.write(gen["graph_response"])
            st.write(f"Generation Time: {gen['graph_generation_time']:.2f} s")


        col_l2, col_r2 = st.columns(2)

        with col_l2:
            st.subheader("Vector Retrieval")
            if vret.get("context"):
                st.write("Context:")
                st.write(vret["context"])
            st.write(f"Top-k: {vret['top_k']}")
            st.write(f"Retrieval Time: {vret['retrieval_time']:.2f} s")

        with col_r2:
            st.subheader("Graph Retrieval")
            if gret.get("context"):
                st.write("Context:")
                st.write(gret["context"])
            st.write(f"Top-k: {gret['top_k']}")
            st.write(f"Retrieval Time: {gret['retrieval_time']:.2f} s")

        col_l3, col_r3 = st.columns(2)

        with col_l3:
            if vret["document_names"]:
                docs_table = pd.DataFrame({
                    "Document Name": vret["document_names"],
                    "Score": vret["distances"],
                    "Document": vret["documents"]
                })
                st.markdown("**Retrieved Documents:**")
                st.dataframe(docs_table, use_container_width=True)

        with col_r3:
            if gret["cypher_query"]:
                st.write("Cypher Query:")
                st.code(gret["cypher_query"])
            if gret['generation_count'] == 0:
                st.write("Generation Count: No generations.")
            else:
                st.write(f"Generation Count: {gret['generation_count']}")
                st.write(f"Generated Cypher: {gret['generated_cypher']}")
                st.write(f"Expanded: {gret['expanded']}, Expansion Success: {gret['expansion_success']}")
            if gret["graph_data"]:
                st.write("Graph Data:")
                st.write(gret["graph_data"])

        st.subheader("Generation Settings")
        st.write(f"Model: {gen['model_name']}")
        st.write(f"Temperature: {gen['temperature']}")
        st.write(f"Max Tokens: {gen['max_tokens']}")
        st.write("Prompt Templates:")
        st.write("User:")
        st.write(f"{gen['prompt']['user_template']}")
        st.write("System:")
        st.write(f"{gen['prompt']['system_template']}")


def _json_serializer(obj):
    from neo4j.time import DateTime
    if isinstance(obj, datetime):
        return obj.isoformat()
    if 'neo4j.time.DateTime' in str(type(obj)):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

class RagReportSaver:
    def __init__(self, reports=None):
        self.reports = reports if reports is not None else []

    def add_report(self, report: RagReport):
        self.reports.append(report)

    def _get_filepath(self, filepath, ext):
        if os.path.isdir(filepath):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_report_{timestamp}.{ext}"
            return os.path.join(filepath, filename)
        if not filepath.lower().endswith(f".{ext}"):
            filepath = f"{filepath}.{ext}"
        return filepath

    def save_json(self, filepath):
        filepath = self._get_filepath(filepath, "json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([r.to_json() for r in self.reports], f, ensure_ascii=False, indent=2, default=_json_serializer)
        return filepath
