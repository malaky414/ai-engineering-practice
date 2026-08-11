import time
import json
import os
from dotenv import load_dotenv
import chromadb
from groq import Groq

from config import (
    KNOWLEDGE_BASE_TEXT, 
    CHUNK_SIZE, 
    CHUNK_OVERLAP, 
    TEST_QUERY, 
    RAG_SYSTEM_PROMPT
)

load_dotenv()


def recursive_chunking(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    separators = ["\n\n", "\n", " ", ""]
    
    def _split(text_block, seps):
        if len(text_block) <= chunk_size or not seps:
            return [text_block]
        sep = seps[0]
        splits = text_block.split(sep) if sep else list(text_block)
        chunks = []
        current_chunk = ""
        for part in splits:
            item = part + sep if sep else part
            if len(current_chunk) + len(item) <= chunk_size:
                current_chunk += item
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = item
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        final_chunks = []
        for c in chunks:
            if len(c) > chunk_size and len(seps) > 1:
                final_chunks.extend(_split(c, seps[1:]))
            else:
                final_chunks.append(c)
        return final_chunks

    return _split(text, separators)


def run_rag_pipeline(query):
    chunks = recursive_chunking(KNOWLEDGE_BASE_TEXT)
    
    client_chroma = chromadb.Client()
    collection = client_chroma.create_collection(name=f"rag_demo_{int(time.time())}")
    ids = [f"chk_{i}" for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks)

    t0_retrieval = time.time()
    retrieved_results = collection.query(query_texts=[query], n_results=3)
    retrieval_latency = time.time() - t0_retrieval
    
    retrieved_chunks = retrieved_results["documents"][0] if retrieved_results["documents"] else []
    context_str = "\n---\n".join(retrieved_chunks)

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    t0_direct = time.time()
    direct_response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": query}],
        temperature=0.0
    ).choices[0].message.content
    direct_latency = time.time() - t0_direct

    rag_prompt = f"{RAG_SYSTEM_PROMPT.format(context=context_str)}\n\nQuestion: {query}"
    t0_rag = time.time()
    rag_response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": rag_prompt}],
        temperature=0.0
    ).choices[0].message.content
    rag_latency = time.time() - t0_rag

    return {
        "query": query,
        "retrieved_context": retrieved_chunks,
        "direct_answer": direct_response,
        "rag_answer": rag_response,
        "metrics": {
            "retrieval_latency_sec": round(retrieval_latency, 4),
            "direct_gen_latency_sec": round(direct_latency, 4),
            "rag_gen_latency_sec": round(rag_latency, 4)
        }
    }


def main():
    print("🚀 Executing End-to-End Dense Retrieval RAG Pipeline...")
    results = run_rag_pipeline(TEST_QUERY)

    with open("rag_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("✅ RAG Pipeline Benchmark complete. Output saved to rag_results.json")


if __name__ == "__main__":
    main()