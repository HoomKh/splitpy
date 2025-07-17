from splitters import splitter
from langchain_community.document_loaders import PyPDFLoader

# file_path = "C:/Users/Hooman/Desktop/langchain-tutorials/tools/nke-10k-2023.pdf"
# loader = PyPDFLoader(file_path=file_path)
# docs = loader.load()

chunks  = splitter.split_text(
    input_data="C:/Users/Hooman/Desktop/langchain-tutorials/tools/test.txt",
    # input_data=docs,
    
    output_type="doc",
    splitter_type="base_character",
    chunk_overlap=50,
    chunk_size=500,
)

#for Doc
# print(f"[✓] Split successful! Total chunks: {len(chunks)}")
# print(f"[+] Type of chunks: {type(chunks)}")
# print(f"[+] Type of first chunk: {type(chunks[0])}")
# print(f"[+] Preview first chunk (page content):\n{'-'*40}\n{chunks[0].page_content[:500]}\n")
# print(f"[+] Preview first chunk (meta data):\n{'-'*40}\n{chunks[0].metadata}")

#for String
print(f"[✓] Split successful! Total chunks: {len(chunks)}")
print(f"[+] Type of chunks: {type(chunks)}")
print(f"[+] Type of first chunk: {type(chunks[0])}")
print(f"[+] Preview first chunk:\n{'-'*40}\n{chunks[0][:500]}")