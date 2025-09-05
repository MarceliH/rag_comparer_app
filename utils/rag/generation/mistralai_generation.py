import os
from mistralai import Mistral
import utils.session_controler as sc

rag_user_prompt_template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context:
{context}

Question:
{query}
"""

rag_system_prompt_template = "You are a helpful assistant."


def get_available_models(mistralai_api_key):
    mistralai_client = Mistral(api_key=mistralai_api_key if mistralai_api_key else os.getenv("MISTRALAI_API_KEY", ""))
    models = mistralai_client.models.list()
    return models.model_dump_json()


def llm_simple_query(mistralai_api_key, query, llm_name):
    mistralai_client = Mistral(api_key=mistralai_api_key if mistralai_api_key else os.getenv("MISTRALAI_API_KEY", ""))

    messages = [
        {
            "role": "user",
            "content": query
        }
    ]

    try:
        chat_response = mistralai_client.chat.complete(
            model=llm_name,
            messages=messages # type: ignore
        )

        return chat_response.choices[0].message.content
    except Exception as e:
        if "Service tier capacity exceeded for this model" in str(e):
            return f"Service tier capacity exceeded for this model."
        else:
            raise e


def llm_rag_query(
    mistralai_api_key,
    query,
    llm_name,
    context=None,
    user_prompt=rag_user_prompt_template,
    system_prompt=rag_system_prompt_template,
    temperature=None,
    max_tokens=None,
    use_temperature=False
):
    """
    Args:
        mistralai_api_key (str): API key for Mistral.
        query (str): User query/question.
        llm_name (str): Name of the LLM model.
        context (str | list[str], optional): Retrieved context snippets.
        temperature (float, optional): Sampling temperature.
        max_tokens (int, optional): Maximum tokens for response.
        use_temperature (bool, optional): Whether to apply temperature.

    Returns:
        str: Model response.
    """

    mistralai_client = Mistral(api_key=mistralai_api_key or os.getenv("MISTRALAI_API_KEY", ""))  

    # Prepare context text
    if context:
        if isinstance(context, list):
            context_text = "\n\n".join(context)
        else:
            context_text = context
    else:
        context_text = "No context provided."

    # Fill in the template
    user_message = user_prompt.format(context=context_text, query=query)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    params = {
        "model": llm_name,
        "messages": messages  # type: ignore
    }
    if use_temperature and temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens

    try:
        chat_response = mistralai_client.chat.complete(**params)
        return chat_response.choices[0].message.content
    except Exception as e:
        error_message = str(e)
        if "Service tier capacity exceeded for this model" in error_message:
            return "Service tier capacity exceeded for this model."
        elif "Invalid model" in error_message or "invalid_model" in error_message:
            return "Invalid model specified."
        else:
            raise e


