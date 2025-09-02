import string
import math
import numpy as np

# --- Sample Documents ---
docs = [
    'information requirement: query considers the user feedback as information requirement to search.',
    'information retrieval: query depends on the model of information retrieval used.',
    'prediction problem: Many problems in information retrieval can be viewed as prediction problems',
    'search: A search engine is one of applications of information retrieval models.',
    'Feedback: feedback is typically used by the system to modify the query and improve prediction',
    'information retrieval: ranking in information retrieval algorithms depends on user query'
]

# --- Preprocessing ---
def clean_text(text):
    return ' '.join(
        ''.join(c for c in word if c not in string.punctuation).lower()
        for word in text.split()
    )

processed_docs = [clean_text(doc) for doc in docs]
vocab = sorted({word for doc in processed_docs for word in doc.split()})
corpus_sets = [set(doc.split()) for doc in processed_docs]

# --- Boolean Model ---
def boolean_model(query):
    query_terms = set(clean_text(query).split())
    relevant = []
    for i, doc_set in enumerate(corpus_sets):
        if query_terms.issubset(doc_set):  # AND semantics
            relevant.append(i)
    return relevant

# --- Vector Space Model ---
def tf(term, doc):
    return doc.count(term)

def idf(term):
    N = len(processed_docs)
    df = sum(1 for d in processed_docs if term in d.split())
    return math.log((N+1) / (df+1)) + 1  # smoothed

def tfidf_vector(text):
    words = text.split()
    return np.array([tf(term, words) * idf(term) for term in vocab])

doc_vectors = [tfidf_vector(doc) for doc in processed_docs]

def cosine_sim(v1, v2):
    num = np.dot(v1, v2)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    return num / denom if denom != 0 else 0

def vector_space_model(query):
    q_vec = tfidf_vector(clean_text(query))
    scores = [(i, cosine_sim(q_vec, d_vec)) for i, d_vec in enumerate(doc_vectors)]
    return [i for i, _ in sorted(scores, key=lambda x: x[1], reverse=True)]

# --- Binary Independence Model (BIM) ---
def df(term):
    return sum(1 for doc in corpus_sets if term in doc)

def rk(term, relevant_ids):
    return sum(1 for i in relevant_ids if term in corpus_sets[i])

def trk(Pk, Qk):
    return math.log10((Pk * (1 - Qk)) / (Qk * (1 - Pk)))

def make_vector(words, vocab):
    return [1 if term in words else 0 for term in vocab]

TD_matrix = [make_vector(doc, vocab) for doc in corpus_sets]

def score(query_vec, weights, matrix):
    q = np.array([q * weights[i] for i, q in enumerate(query_vec)])
    scores = []
    for i, doc in enumerate(matrix):
        d = np.array([v * weights[j] for j, v in enumerate(doc)])
        scores.append((i, np.dot(d, q)))
    return sorted(scores, key=lambda x: x[1], reverse=True)

def phase1(query_vec):
    weights = []
    for term in vocab:
        Pk = 0.5
        Qk = (df(term) + 0.5) / (len(corpus_sets) + 1)
        weights.append(trk(Pk, Qk))
    ranked = score(query_vec, weights, TD_matrix)
    return [i for i, _ in ranked[:3]]  # assume top-3 relevant

def phase2(query_vec, relevant_ids):
    weights = []
    Nr = len(relevant_ids)
    for term in vocab:
        Rk = rk(term, relevant_ids)
        Pk = (Rk + 0.5) / (Nr + 1)
        Qk = (df(term) - Rk + 0.5) / (len(corpus_sets) - Nr + 1)
        weights.append(trk(Pk, Qk))
    ranked = score(query_vec, weights, TD_matrix)
    return [i for i, _ in ranked]

def BIM(query):
    query_set = set(clean_text(query).split())
    query_vec = make_vector(query_set, vocab)
    rel1 = phase1(query_vec)
    rel2 = phase2(query_vec, rel1)
    return rel2

# --- Evaluation Metrics ---
def evaluate(retrieved, relevant):
    retrieved_set, relevant_set = set(retrieved), set(relevant)
    true_pos = len(retrieved_set & relevant_set)
    precision = true_pos / len(retrieved) if retrieved else 0
    recall = true_pos / len(relevant) if relevant else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision+recall) else 0
    return precision, recall, f1

 
if __name__ == "__main__":
    query = "information retrieval query"

    # Ground truth (manual relevance for testing) 
    relevant_docs = [1, 2, 5]

    print("\nBoolean Model")
    bm_results = boolean_model(query)
    print("Retrieved:", bm_results)
    p, r, f1 = evaluate(bm_results, relevant_docs)
    print(f"Precision={p:.2f}, Recall={r:.2f}, F1={f1:.2f}")

    print("\nVector Space Model")
    vsm_results = vector_space_model(query)
    print("Ranking:", vsm_results)
    p, r, f1 = evaluate(vsm_results[:3], relevant_docs)  # top-3 for fair compare
    print(f"Precision={p:.2f}, Recall={r:.2f}, F1={f1:.2f}")

    print("\nBinary Independence Model")
    bim_results = BIM(query)
    print("Ranking:", bim_results)
    p, r, f1 = evaluate(bim_results[:3], relevant_docs)  # top-3
    print(f"Precision={p:.2f}, Recall={r:.2f}, F1={f1:.2f}")
