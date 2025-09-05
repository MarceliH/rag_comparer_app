from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def recursive_text_splitter(text: str, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # length_function=len,
        # is_separator_regex=False,
    )

    splitted_text = text_splitter.split_text(text=text)
    return splitted_text
