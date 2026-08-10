import time
import json
import re
import numpy as np
import chromadb
from config import SAMPLE_DOCUMENT, CHUNK_SIZE, CHUNK_OVERLAP, TEST_QUERY


def fixed_size_chunking(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks


def sentence_chunking(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]


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


def evaluate_chunking_retrieval(strategy_name, chunks, query):
    client = chromadb.Client()
    collection = client.create_collection(name=f"bench_{strategy_name}_{int(time.time()*1000)}")

    t0 = time.time()
    ids = [f"chk_{i}" for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks)
    indexing_time = time.time() - t0

    t1 = time.time()
    results = collection.query(query_texts=[query], n_results=1)
    query_time = time.time() - t1

    top_doc = results["documents"][0][0] if results["documents"] else ""
    distance = results["distances"][0][0] if results["distances"] else 0.0
    similarity_score = round(1.0 / (1.0 + distance), 4)

    return {
        "num_chunks": len(chunks),
        "avg_chunk_length": round(float(np.mean([len(c) for c in chunks])), 1) if chunks else 0,
        "indexing_time_seconds": round(indexing_time, 5),
        "query_time_seconds": round(query_time, 5),
        "top_similarity_score": similarity_score,
        "top_retrieved_chunk": top_doc[:80] + "..." if len(top_doc) > 80 else top_doc
    }


def main():
    print("✂️ Running Chunking Strategies Benchmark...")

    fixed_chunks = fixed_size_chunking(SAMPLE_DOCUMENT)
    fixed_eval = evaluate_chunking_retrieval("fixed", fixed_chunks, TEST_QUERY)

    sentence_chunks = sentence_chunking(SAMPLE_DOCUMENT)
    sentence_eval = evaluate_chunking_retrieval("sentence", sentence_chunks, TEST_QUERY)

    recursive_chunks = recursive_chunking(SAMPLE_DOCUMENT)
    recursive_eval = evaluate_chunking_retrieval("recursive", recursive_chunks, TEST_QUERY)

    payload = {
        "test_query": TEST_QUERY,
        "results": {
            "Fixed-Size": fixed_eval,
            "Sentence-Based": sentence_eval,
            "Recursive-Character": recursive_eval
        }
    }

    with open("chunking_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("✅ Benchmark complete. Metrics saved to chunking_results.json")


if __name__ == "__main__":
    main()