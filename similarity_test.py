from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3')
pairs=[
    ("How do I reset my password?", "What's the process to change my password?"),
    ("How do I reset my password?", "How many vacation days do I get?"),
    ("The meeting is at 3 PM tomorrow.", "Tomorrow's meeting starts at 15:00."),
    ("The meeting is at 3 PM tomorrow.", "I like pizza with extra cheese."),
    ("كم عدد أيام الإجازة السنوية؟", "How many annual leave days are there?"),
]
if __name__ == "__main__":
    for sentence_a , sentence_b in pairs:
        embedding_a = model.encode(sentence_a)
        embedding_b = model.encode(sentence_b)
        similarity_score = model.similarity(embedding_a, embedding_b)
        print(f"Sentence A: {sentence_a}")
        print(f"Sentence B: {sentence_b}")
        print(f"Similarity Score: {similarity_score.item():.4f}\n")