SAMPLE_DOCUMENT = """
Artificial Intelligence and Machine Learning have transformed modern software engineering. 
In a Retrieval-Augmented Generation (RAG) architecture, long documents are divided into smaller pieces called chunks. 
Chunking is a critical step because Large Language Models (LLMs) have finite context windows and processing limits.

Fixed-size chunking splits text strictly by character or token count, regardless of semantic structure. 
While simple and fast, it often splits sentences in the middle, destroying semantic context and reducing retrieval quality.

Sentence-based chunking respects punctuation marks such as periods and newlines. 
This preserves full logical statements, but chunk sizes can vary dramatically depending on author writing style.

Recursive character chunking attempts to split text using a hierarchy of separators like paragraphs, sentences, and words. 
It aims to keep related pieces together while maintaining a constrained maximum chunk size with optional overlapping.

Vector databases like ChromaDB store these chunks as dense vector embeddings. 
When a user issues a search query, the system computes similarity between query embedding and chunk embeddings to retrieve context.
"""

CHUNK_SIZE = 150
CHUNK_OVERLAP = 30
TEST_QUERY = "Why is fixed-size chunking problematic for context?"