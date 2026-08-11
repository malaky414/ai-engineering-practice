KNOWLEDGE_BASE_TEXT = """
Atlas Enterprise Infrastructure Deployment Guidelines (2026 Edition):
1. Security Protocols: All production microservices must communicate using TLS 1.3 encryption. Internal endpoints without valid mTLS certificates will be terminated automatically by the ingress controller after 3 failed attempts.
2. Vector Database Allocation: Production RAG pipelines must utilize Qdrant or ChromaDB with HNSW indexing. The maximum allowed memory footprint per instance is 16GB RAM.
3. Model Inference Timeouts: LLM API requests routed through the primary gateway have a hard timeout limit of 5.0 seconds. If a model exceeds this threshold, the request falls back to the local LLaMA-3.1-8B-Instant instance.
4. Data Retention: Customer query logs and conversation embeddings are retained for exactly 90 days in cold storage before automated permanent erasure.
"""

CHUNK_SIZE = 350
CHUNK_OVERLAP = 50

TEST_QUERY = "What happens if an internal endpoint fails mTLS authentication three times?"

RAG_SYSTEM_PROMPT = """You are an accurate technical assistant. 
Answer the user question strictly using ONLY the provided context below. 
If the answer cannot be found in the context, say 'Information not found in context.'

Context:
{context}
"""