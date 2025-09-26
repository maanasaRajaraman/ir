# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 22:27:31 2025

@author: maana
"""

# q1 - svd
# q2 - page rank : binary matrix, eigenval
# q3 - minhash

# SVD with Latent Semantic Indexing (LSI)
import random
import numpy as np
from numpy.linalg import svd
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity


docs = [
    "the cat sat on the mat",
    "the dog sat on the mat",
    "the cat chased the dog"
]

# svd
vocab = sorted(set(" ".join(docs).split()))
A = np.array([[doc.split().count(w) for doc in docs] for w in vocab])
print("Term frequency Matrix : ")
print(A)

U, S, Vt = svd(A, full_matrices=False)
print("Singular Values : ", S)

# Document Embeddings
k = 2
U_k, S_k, Vt_k = U[:, :k], np.diag(S[:k]), Vt[:k, :]

document_embeddings = (S_k @ Vt_k).T
print("Document Embeddings : ")
print(document_embeddings)

query = "cat on mat"
query_vec = np.array([query.split().count(word) for word in vocab])
query_embedding = query_vec @ U_k @ np.linalg.inv(S_k)

sims = cosine_similarity([query_embedding], document_embeddings)[0]
print("Cosine similarity of query wrt to Docs: ")
for i, d in enumerate(sims):
    print(f"Doc {i+1} has similarity {sims[i]}")

print("--------------------------------")
# page rank
def pagerank_power_iteration(adj_matrix, alpha = 0.86, tol=1e-6, max_iter = 100):
    n = adj_matrix.shape[0]
    H = np.zeros((n,n))
    for i in range(n):
        outlinks = np.sum(adj_matrix[i])
        if outlinks > 0:
            H[i] = adj_matrix[i]/outlinks
    for i in range(n):
        if np.sum(H[i])==0:
            H[i] = np.ones(n)/n
    
    S = H.copy()
    G = alpha*S + (1-alpha)*(np.ones((n,n))/n)
    rank = np.ones(n)/n
    for _ in range(max_iter):
        new_rank = rank @ G
        if np.linalg.norm(new_rank - rank, 1) < tol:
             break
        rank = new_rank
    return rank

def pagerank_eigenValues(adj_matrix, alpha = 0.86, tol=1e-6, max_iter = 100):
    n = adj_matrix.shape[0]
    H = np.zeros((n,n))
    for i in range(n):
        outlinks = np.sum(adj_matrix[i])
        if outlinks > 0:
            H[i] = adj_matrix[i]/outlinks
    for i in range(n):
        if np.sum(H[i])==0:
            H[i] = np.ones(n)/n
    
    S = H.copy()
    G = alpha*S + (1-alpha)*(np.ones((n,n))/n)
    rank = np.ones(n)/n
    
    eigenval, eigenvecs = np.linalg.eig(G.T)
    idx = np.argmin(np.abs(eigenval-1))
    principal_vector = np.real(eigenvecs[:, idx])
    
    pagerank_scores = [principal_vector[i] /np.sum(principal_vector) for i in range(n)]
    return pagerank_scores

adj_matrix = np.array([
    [0, 1, 0, 0, 0],  # Node 1
    [1, 0, 0, 0, 1],  # Node 2
    [1, 1, 0, 1, 1],  # Node 3
    [0, 0, 0, 0, 1],  # Node 4
    [0, 0, 0, 1, 0]
    ])

print("page rank via power iteration: ")
print(pagerank_power_iteration(adj_matrix))

print("page rank via eigen values: ")
print(pagerank_eigenValues(adj_matrix))

print("--------------------------------")

# min hash : shingles

def get_shingles(docs, k):
    words = docs.split()
    return {" ".join(words[i:i+k]) for i in range(len(words)-k+1)}

k = 2
doc_shingles = [get_shingles(d, k) for d in docs]
shingles = sorted(set().union(*doc_shingles))

sd_matrix = np.array([[int(sh in ds) for ds in doc_shingles] for sh in shingles])
print(sd_matrix)

num_shingles, num_docs = sd_matrix.shape
num_hash = 5
signature = np.full((num_hash, num_docs), np.inf)

row = list(range(num_shingles))
permutation = [random.sample(row, len(row)) for i in range(num_hash)]

for i, perm in enumerate(permutation):
    for col in range(num_docs):
        for row in perm:
            if sd_matrix[row, col]:
                signature[i, col] = row
                break
print("Min hash with shingles : ")            
print("signature matrix: ")
print(signature)
def minhash_sim(col1, col2):
    return np.mean(signature[:, col1] == signature[:, col2])

print("\nSimilarity between Doc1 & Doc2:", minhash_sim(0, 1))
print("Similarity between Doc1 & Doc3:", minhash_sim(0, 2))
print("Similarity between Doc2 & Doc3:", minhash_sim(1, 2))

print("--------------------------------")

# min hash : cosine and euclidean similarity
vocab = sorted(set(" ".join(docs).split()))
tf = np.array([[doc.split().count(w) for w in vocab] for doc in docs])
N = len(docs)
idf = np.log(N/ np.count_nonzero(tf, axis=0))

tfidf = tf*idf
print("tf-idf matrix: \n", tfidf)

print("Cosine Similarity: ")
for i in range(0, N):
    for j in range(i+1, N):
        cosSimilarity = np.dot(tfidf[i], tfidf[j]) / (norm(tfidf[i]) * norm(tfidf[j]))
        print(f"Cosine Similarity of {i+1} and {j+1} = {cosSimilarity:0.4f}")
    
print("Euclidean Similarity : ")
for i in range(0, N):
    for j in range(i+1, N):
        euc = (norm(tfidf[i]- tfidf[j]))
        print(f"Euclidean Dis of {i+1} and {j+1} = {cosSimilarity:0.4f}")
    