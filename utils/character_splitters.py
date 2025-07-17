# ----------------------- ( character-based ) -----------------------
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from typing import Union, Literal


def character_text_splitter(
    input_data: Union[str, list[Document], list[str]],
    output_type: Literal["document", "string"],
    separator: list[str],
    chunk_size: int,
    chunk_overlap: int,
    length_function,
    is_separator_regex: bool,
) -> Union[list[Document], list[str]]:
    
    from .helper import normalize_to_documents, final_decision_output

    # Initialize
    splitter = CharacterTextSplitter(
        separator=separator[0],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        is_separator_regex=is_separator_regex,
    )

    # Convert all input to list[Document]
    docs = normalize_to_documents(input_data=input_data)

    # Apply splitter
    split_docs = splitter.split_documents(docs)

    # Return format output
    return final_decision_output(split_docs=split_docs, output_type=output_type)


# ----------------------- ( Character Text-structured based ) -----------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter


def recursive_character_text_splitter(
    input_data: Union[str, list[Document], list[str]],
    output_type: Literal["document", "string"],
    chunk_size: int,
    chunk_overlap: int,
    length_function,
    separators: list[str],
    is_separator_regex: bool,
    add_start_index: bool,
) -> Union[list[Document], str]:
    from .helper import normalize_to_documents, final_decision_output

    # Initialize
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        is_separator_regex=is_separator_regex,
        add_start_index=add_start_index,
        separators=separators,
    )

    # Convert all input to list[Document]
    docs = normalize_to_documents(input_data=input_data)

    # Apply splitter
    split_docs = splitter.split_documents(docs)

    # Return format output
    return final_decision_output(split_docs=split_docs, output_type=output_type)
