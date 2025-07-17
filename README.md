# SplitPy

# 📚 LangChain Text Splitters Toolkit

This module provides various document splitting strategies for preparing text to be processed by LLMs or NLP systems.

---

## ✨ Why split documents?

There are several practical reasons for splitting text into smaller, manageable chunks:

* ✅ Avoiding input length limits in LLMs (like OpenAI models)
* ✅ Improving semantic representation for long documents
* ✅ Enhancing search precision in RAG and embedding systems
* ✅ Making document processing more memory and compute efficient
* ✅ Enabling context windowing in chat systems and pipelines

---

## 🧠 Strategies included

| Splitter                                     | Type      | Token-aware | Recursive | Description                         |
| -------------------------------------------- | --------- | ----------- | --------- | ----------------------------------- |
| `character_text_splitter`                    | character | ❌           | ❌         | Basic fixed-size character chunks   |
| `recursive_character_text_splitter`          | character | ❌           | ✅         | Smarter structure-aware chunking    |
| `tiktoken_character_text_splitter`           | token     | ✅           | ❌         | Token-limited LLM-safe splitting    |
| `tiktoken_recursive_character_text_splitter` | token     | ✅           | ✅         | Best for OpenAI-compatible RAG      |
| `tiktoken_token_text_splitter`               | token     | ✅           | ❌         | Raw token chunking (fast but blind) |

---

## ⚙️ Example usage

```python
from tools.splitters import tiktoken_recursive_character_text_splitter

chunks = tiktoken_recursive_character_text_splitter(
    path="data/article.txt",
    chunk_size=200,
    chunk_overlap=30,
    encoding_name="cl100k_base"
)

print(chunks[0].page_content)
```

---

## 🧪 When to use which?

| Use Case                     | Recommended Splitter                         |
| ---------------------------- | -------------------------------------------- |
| RAG + OpenAI                 | `tiktoken_recursive_character_text_splitter` |
| Basic NLP preprocessing      | `character_text_splitter`                    |
| Fast token slicing           | `tiktoken_token_text_splitter`               |
| Semantic chunking for search | `recursive_character_text_splitter`          |

---

## 📌 Source reference

This implementation follows concepts from LangChain’s official docs:
🔗 [LangChain Docs: Text Splitters](https://python.langchain.com/docs/concepts/text_splitters/)
