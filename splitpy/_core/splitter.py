from pathlib import Path
from typing import Union, Callable
from langchain_core.documents import Document
from . import (
    OutputType, 
    SplitterType,
    router_splitter_type,
    load_config,
    resolve_length_function,
)


# Import default config from config.yml

default_config = load_config()
if isinstance(default_config.get("length_function"), str):
    default_config["length_function"] = resolve_length_function(
        default_config["length_function"]
    )


def split_text(
    input_data: Union[str, Path, list[str], list[Document]] = default_config[
        "input_data"
    ],
    output_type: OutputType = default_config["output_type"],
    splitter_type: SplitterType = default_config["splitter_type"],
    # Encoding for loading .txt file
    encoding: str = default_config["encoding"],
    # Parameters for Chunk
    chunk_size: int = default_config["chunk_size"],
    chunk_overlap: int = default_config["chunk_overlap"],
    encoding_name: str = default_config["encoding_name"],
    length_function: Callable = default_config["length_function"],
    is_separator_regex: bool = default_config["is_separator_regex"],
    add_start_index: bool = default_config["add_start_index"],
    separator: list[str] = default_config["separator"],
) -> Union[list[Document], list[str]]:

    return router_splitter_type(
        input_data=input_data,
        output_type=output_type,
        splitter_type=splitter_type,
        encoding=encoding,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name=encoding_name,
        length_function=length_function,
        is_separator_regex=is_separator_regex,
        add_start_index=add_start_index,
        separator=separator,
    )
