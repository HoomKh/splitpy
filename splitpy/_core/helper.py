from langchain_core.documents import Document
from typing import Callable, Union
from pathlib import Path
from . import (
    OutputType, 
    SplitterType, 
    character_text_splitter,
    recursive_character_text_splitter,
    tiktoken_character_text_splitter,
    tiktoken_recursive_character_text_splitter,
    tiktoken_token_text_splitter,
)


def router_splitter_type(
    input_data: Union[str, Path, list[str], list[Document]],
    output_type: OutputType,
    splitter_type: SplitterType,
    encoding: str,
    # Parameters for Chunk
    chunk_size: int,
    chunk_overlap: int,
    encoding_name: str,
    length_function: Callable,
    is_separator_regex: bool,
    add_start_index: bool,
    separator: list[str],
) -> Union[list[Document], list[str]]:

    # Load the input_data if it's .txt -> str
    if isinstance(input_data, Path) or (
        isinstance(input_data, str) and input_data.endswith(".txt")
    ):
        path = Path(input_data)
        input_data = load_txt(path, encoding=encoding)

    # Basic Charecter Text Splitter
    if splitter_type == SplitterType.BASE_CHARACTER:
        return character_text_splitter(
            input_data=input_data,
            output_type=output_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
            length_function=length_function,
            is_separator_regex=is_separator_regex,
        )

    # Recursive Character Text Splitter
    elif splitter_type == SplitterType.BASE_RECURSIVE:
        return recursive_character_text_splitter(
            input_data=input_data,
            output_type=output_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separator,
            length_function=length_function,
            is_separator_regex=is_separator_regex,
            add_start_index=add_start_index,
        )

    # Tiktoken Token Text Splitter
    elif splitter_type == SplitterType.TIKTOKEN_TOKEN:
        return tiktoken_token_text_splitter(
            input_data=input_data,
            output_type=output_type,
            encoding_name=encoding_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # Tiktoken Character Text Splitter
    elif splitter_type == SplitterType.TIKTOKEN_CHARACTER:
        return tiktoken_character_text_splitter(
            input_data=input_data,
            output_type=output_type,
            encoding_name=encoding_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # Tiktoken Recursive Text Splitter
    elif splitter_type == SplitterType.TIKTOKEN_RECURSIVE:
        return tiktoken_recursive_character_text_splitter(
            input_data=input_data,
            output_type=output_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name,
        )
    else:
        raise ValueError(
            f"Splitter type '{splitter_type}' does not support path input directly"
        )


def normalize_to_documents(input_data) -> list[Document]:
    if isinstance(input_data, str):
        return [Document(page_content=input_data)]
    elif isinstance(input_data, list) and all(isinstance(i, str) for i in input_data):
        return [Document(page_content=i) for i in input_data]
    elif isinstance(input_data, list) and all(
        isinstance(i, Document) for i in input_data
    ):
        return input_data
    else:
        raise ValueError("Unsupported input type")


def load_txt(path: str, encoding: str) -> str:
    try:
        with open(path, encoding=encoding) as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"error while reading file : {e}")


def final_decision_output(
    split_docs: list[Document], output_type: OutputType
) -> Union[list[Document], list[str]]:

    if output_type == OutputType.DOCUMENT:
        return split_docs
    elif output_type == OutputType.STRING:
        return [s.page_content for s in split_docs]
    else:
        raise ValueError("output_type must be 'document' or 'string'")


def load_config() -> dict:
    return {
        "input_data": None,
        "output_type":  OutputType.DOCUMENT,
        "splitter_type": SplitterType.BASE_RECURSIVE,
        "encoding": "utf-8",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "encoding_name": "cl100k_base",
        "length_function": "len",
        "is_separator_regex": False,
        "add_start_index": False,
        "separator": [
            "\n",
            "\n\n",
            " ",
            ".",
            ",",
            "\u200b",
            "\uff0c",
            "\u3001",
            "\uff0e",
            "\u3002",
            "",
        ]
    }


def resolve_length_function(func_name: str) -> Callable:
    if func_name == "len":
        return len
    raise ValueError(f"Unsupported length function: {func_name}")
