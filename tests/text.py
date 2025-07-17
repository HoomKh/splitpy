from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from _core import split_text, SplitterType, OutputType

file_path = "C:/Users/Hooman/Desktop/splitpy/tests/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path=file_path)
docs = loader.load()

chunks  = split_text(
    # input_data="C:/Users/Hooman/Desktop/langchain-tutorials/tools/test.txt",
    input_data=docs,
    
    output_type=OutputType.DOCUMENT,
    splitter_type=SplitterType.BASE_CHARACTER,
    chunk_overlap=50,
    chunk_size=500,
)

#for Doc
print(f"[✓] Split successful! Total chunks: {len(chunks)}")
print(f"[+] Type of chunks: {type(chunks)}")
print(f"[+] Type of first chunk: {type(chunks[0])}")
print(f"[+] Preview first chunk (page content):\n{'-'*40}\n{chunks[0].page_content[:500]}\n")
print(f"[+] Preview first chunk (meta data):\n{'-'*40}\n{chunks[0].metadata}")

#for String
# print(f"[✓] Split successful! Total chunks: {len(chunks)}")
# print(f"[+] Type of chunks: {type(chunks)}")
# print(f"[+] Type of first chunk: {type(chunks[0])}")
# print(f"[+] Preview first chunk:\n{'-'*40}\n{chunks[0][:500]}")