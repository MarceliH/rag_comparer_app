import time
import os
import re
from langchain_neo4j import Neo4jGraph
from langchain_mistralai import ChatMistralAI
from pydantic import SecretStr
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from neo4j_graphrag.schema import format_schema
from neo4j_graphrag.retrievers.text2cypher import extract_cypher
from langchain_huggingface import HuggingFaceEmbeddings
import time



cypher_generation_prompt_template = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""

cypher_generation_prompt_template_alt = """Task: Generate a Cypher statement to query a graph database.

Instructions:
- Use only the relationship types and properties provided in the schema.
- Assign a variable name to every node you use in MATCH patterns.
- RETURN the nodes themselves if possible, not just their properties, to allow graph traversal.
- Do not include explanations or any text outside the Cypher statement.

Schema:
{schema}

Question:
{question}"""


def instantiate_neo4j_graph(uri: str | None, username: str | None, password: str | None) -> Neo4jGraph | None:
    try:
        return Neo4jGraph(url=uri, username=username, password=password, sanitize=True, refresh_schema=True) # langchain function already handles None values with checking environment variables
    except Exception as e:
        print(f"Error instantiating Neo4jGraph: {e}")
        return None


def expand_query(cypher: str, depth: int = 2, top_k: int = 50) -> str | None:
    # Remove RETURN and ORDER and LIMIT
    cypher_cleaned = re.sub(r"(?i)\bRETURN\b.+?(?=(?:\bORDER\s+BY\b|\bLIMIT\b|$))", "", cypher, flags=re.DOTALL)
    cypher_cleaned = re.sub(r"(?i)\bLIMIT\b\s*\d+", "", cypher_cleaned)

    # Get nodes from RETURN
    m = re.search(r"(?i)\bRETURN\b(.+?)(?:(?:\bORDER\s+BY\b)|\bLIMIT\b|$)", cypher, flags=re.DOTALL)
    node_vars = []
    if m:
        node_vars = [part.strip().split(" ")[0] for part in m.group(1).split(",") if "." not in part and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", part.strip().split(" ")[0])]
    if not node_vars:
        node_vars = list(set(re.findall(r"\(([A-Za-z_][A-Za-z0-9_]*)[:)]", cypher)))
        if not node_vars:
            return None

    call_return = ", ".join(f"{v} AS {v}_seed" for v in node_vars)
    keep_vars_csv = ", ".join(f"{v}_seed" for v in node_vars)

    wrapped = f"""
    CALL {{
      {cypher_cleaned}
      RETURN {call_return}
    }}
    WITH {keep_vars_csv}
    WITH [x IN [{keep_vars_csv}] WHERE NOT "Chunk" IN labels(x)] AS seeds
    UNWIND seeds AS seed
    MATCH path = (seed)-[*1..{depth}]-(n)
    WITH nodes(path) AS path_nodes, relationships(path) AS path_rels
    RETURN
      [node IN path_nodes |
         {{value: CASE
                   WHEN "__Entity__" IN labels(node) THEN node.id
                   WHEN "Chunk" IN labels(node) THEN "[Chunk node]"
                   ELSE null
                 END
         }}
      ] AS nodes_info,
      [rel IN path_rels |
         {{type: type(rel)}}
      ] AS rels_info
    LIMIT {top_k}
    """
    return wrapped.strip()


def format_graph_result(result: dict) -> str:
    nodes = [n.get("value") for n in result.get("nodes_info", []) if n.get("value")]
    rels = [r.get("type") for r in result.get("rels_info", []) if r.get("type")]

    path_parts = []
    for i, node in enumerate(nodes):
        path_parts.append(str(node))
        if i < len(rels):
            path_parts.append(f"--{rels[i]}-->")

    return " ".join(path_parts)


def neo4j_text2cypher_retriever(
    graph: Neo4jGraph,
    llm_model: str | None,
    llm_key: str | None,
    query_text: str,
    top_k: int = 5,
    cypher_generation_prompt: str = cypher_generation_prompt_template,
    max_generations: int = 1,
    expand: bool = False,
    traversal_depth: int = 2,
    
):
    llm = ChatMistralAI(
        name=llm_model,
        api_key=SecretStr(llm_key or os.getenv("MISTRALAI_API_KEY", "")),
    )

    cypher_generation_prompt_query = PromptTemplate(
        input_variables=["schema", "question"], template=cypher_generation_prompt
    )

    cypher_generation_chain = (
        cypher_generation_prompt_query | llm | StrOutputParser()
    )

    raw_schema = format_schema(graph.get_structured_schema, is_enhanced=False)
    graph_schema = re.sub(r', embedding: LIST', '', raw_schema) # remove unnecessary for cypher generation embedding property

    args = {
        "question": query_text,
        "schema": graph_schema,
    }

    
    generation_count = 0
    graph_data = []
    generated_cypher = ""
    expansion_success = False

    start_time = time.time()
    for attempt in range(max_generations):
        generation_count += 1
        try:
            generated_cypher = cypher_generation_chain.invoke(args)
            generated_cypher = extract_cypher(generated_cypher)
        except Exception as e:
            generated_cypher = ""
            continue

        if not generated_cypher:
            continue

        try:
            if expand:
                expanded_query = expand_query(generated_cypher, depth=traversal_depth, top_k=top_k)
                if expanded_query:
                    try:
                        graph_data = graph.query(expanded_query)
                        expansion_success = True
                        if not graph_data:
                            graph_data = graph.query(generated_cypher)[:top_k]
                            expansion_success = False
                    except Exception:
                        graph_data = graph.query(generated_cypher)[:top_k]
                        expansion_success = False
                else:
                    graph_data = graph.query(generated_cypher)[:top_k]
                    expansion_success = False
            else:
                graph_data = graph.query(generated_cypher)[:top_k]
                expansion_success = False
        except Exception as e:
            graph_data = []
            expansion_success = False
            continue

        if graph_data:
            break

    retrieval_time = time.time() - start_time

    formatted_context = "\n".join(format_graph_result(r) for r in graph_data)
    return {
        "context": formatted_context,
        "graph_data": graph_data,
        "generated_cypher": generated_cypher,
        "expanded": expand,
        "expansion_success": expansion_success,
        "cypher_query": expanded_query if expand and expansion_success else generated_cypher,
        "generation_count": generation_count,
        "retrieval_time": retrieval_time,
        "top_k": top_k,
    }


def format_graph_result_vector(result: dict) -> str:
    """Format graph paths for readability."""
    nodes = [str(n) for n in result.get("path_nodes", [])]
    rels = result.get("path_rels", [])

    path_parts = []
    for i, node in enumerate(nodes):
        path_parts.append(node)
        if i < len(rels):
            path_parts.append(f"--{rels[i]}-->")
    return " ".join(path_parts)


cypher_traversal_template = """
WITH $query_embedding AS queryEmbedding
MATCH (n)
WHERE n.embedding IS NOT NULL
WITH n, gds.similarity.cosine(n.embedding, queryEmbedding) AS score
ORDER BY score DESC
LIMIT $top_k
CALL {{
  WITH n
  MATCH path = (n)-[*1..{depth}]-(m)
  WITH nodes(path) AS path_nodes, relationships(path) AS path_rels
  RETURN
    [node IN path_nodes |
      CASE
        WHEN "__Entity__" IN labels(node) THEN node.id
        WHEN "Chunk" IN labels(node) THEN "[chunk]"
        ELSE node
      END
    ] AS path_nodes,
    [rel IN path_rels | type(rel)] AS path_rels
}}
RETURN n, score, path_nodes, path_rels
"""


def neo4j_vector_retriever(
    graph: Neo4jGraph,
    query_text: str,
    top_k: int = 5,
    traversal_depth: int = 2,
    cypher_query: str = cypher_traversal_template,
):
    start_time = time.time()

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = embeddings.embed_query(query_text)

    cypher_query_final = cypher_query.format(depth=traversal_depth)

    results = graph.query(
        cypher_query_final,
        params={"query_embedding": query_embedding, "top_k": top_k}
    )

    retrieval_time = time.time() - start_time

    formatted_context = "\n".join(format_graph_result_vector(r) for r in results)

    return {
        "context": formatted_context,
        "graph_data": results,
        "generated_cypher": None,
        "expanded": True,
        "expansion_success": bool(results),
        "cypher_query": cypher_query_final,
        "generation_count": 0,
        "retrieval_time": retrieval_time,
        "top_k": top_k,
    }
