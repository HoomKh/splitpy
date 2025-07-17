# ----------------------- tiktoken ( Token-based ) -----------------------
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_core.documents import Document
from typing import Union
from . import OutputType


def tiktoken_character_text_splitter(
    input_data: Union[str, list[Document], list[str]],
    output_type: OutputType,
    encoding_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Union[list[Document], list[str]]:
    from .helper import normalize_to_documents, final_decision_output

    # Initialize
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Convert all input to list[Document]
    docs = normalize_to_documents(input_data=input_data)

    # Apply splitter
    split_docs = splitter.split_documents(docs)

    # Return format output
    return final_decision_output(split_docs=split_docs, output_type=output_type)


def tiktoken_recursive_character_text_splitter(
    input_data: Union[str, list[Document], list[str]],
    output_type: OutputType,
    encoding_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Union[list[Document], list[str]]:
    from .helper import normalize_to_documents, final_decision_output

    # Initialize
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Convert all input to list[Document]
    docs = normalize_to_documents(input_data=input_data)

    # Apply splitter
    split_docs = splitter.split_documents(docs)

    # Return format output
    return final_decision_output(split_docs=split_docs, output_type=output_type)


def tiktoken_token_text_splitter(
    input_data: Union[str, list[Document], list[str]],
    output_type: OutputType,
    encoding_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Union[list[Document], list[str]]:
    from .helper import normalize_to_documents, final_decision_output

    # Initialize
    splitter = TokenTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, encoding_name=encoding_name
    )

    # Convert all input to list[Document]
    docs = normalize_to_documents(input_data=input_data)

    # Apply splitter
    split_docs = splitter.split_documents(docs)

    # Return format output
    return final_decision_output(split_docs=split_docs, output_type=output_type)
