# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 12:18:33 2025

@author: preth
"""

import math
from collections import Counter

class VectorSpaceModel:
    def __init__(self, docs):
        self.docs = docs
        self.vocab, self.tf_matrix = self._preprocess()
        self.idf = self._compute_idf()
        self.tfidf_matrix = self._compute_tfidf()
    
    def _preprocess(self):
        """Integrated preprocessing"""
        processed = []
        vocab_set = set()
        
        for doc in self.docs:
            tokens = doc.lower().split()
            processed.append(tokens)
            vocab_set.update(tokens)
        
        vocab = sorted(vocab_set)
        
        # Build TF matrix
        tf_matrix = []
        for doc_tokens in processed:
            counts = Counter(doc_tokens)
            tf_row = [counts.get(term, 0) for term in vocab]
            tf_matrix.append(tf_row)
        
        return vocab, tf_matrix
    
    def _compute_idf(self):
        """IDF calculation"""
        N = len(self.tf_matrix)
        idf = []
        for term_idx in range(len(self.vocab)):
            df = sum(1 for doc in self.tf_matrix if doc[term_idx] > 0)
            idf.append(math.log(N / df) if df > 0 else 0)
        return idf
    
    def _compute_tfidf(self):
        """TF-IDF matrix"""
        return [[tf * self.idf[i] for i, tf in enumerate(doc)] 
                for doc in self.tf_matrix]
    
    def _query_to_vector(self, query):
        """Query to TF-IDF vector"""
        query_tf = Counter(query.lower().split())
        return [query_tf.get(term, 0) * self.idf[i] 
                for i, term in enumerate(self.vocab)]
    
    def cosine_similarity(self, v1, v2):
        """Cosine similarity"""
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a * a for a in v1))
        mag2 = math.sqrt(sum(a * a for a in v2))
        return dot / (mag1 * mag2) if mag1 and mag2 else 0
    
    def jaccard_coefficient(self, v1, v2):
        """Jaccard coefficient for binary vectors"""
        # Convert to binary
        b1 = [1 if x > 0 else 0 for x in v1]
        b2 = [1 if x > 0 else 0 for x in v2]
        
        intersection = sum(a & b for a, b in zip(b1, b2))
        union = sum(a | b for a, b in zip(b1, b2))
        
        return intersection / union if union > 0 else 0
    
    def dice_coefficient(self, v1, v2):
        """Dice coefficient"""
        b1 = [1 if x > 0 else 0 for x in v1]
        b2 = [1 if x > 0 else 0 for x in v2]
        
        intersection = sum(a & b for a, b in zip(b1, b2))
        total = sum(b1) + sum(b2)
        
        return (2 * intersection) / total if total > 0 else 0
    
    def dot_product(self, v1, v2):
        """Simple dot product"""
        return sum(a * b for a, b in zip(v1, v2))
    
    def search(self, query, similarity='cosine', top_k=5):
        """Search with different similarity measures"""
        qvec = self._query_to_vector(query)
        
        similarities = []
        for doc_id, dvec in enumerate(self.tfidf_matrix):
            if similarity == 'cosine':
                sim = self.cosine_similarity(qvec, dvec)
            elif similarity == 'jaccard':
                sim = self.jaccard_coefficient(qvec, dvec)
            elif similarity == 'dice':
                sim = self.dice_coefficient(qvec, dvec)
            elif similarity == 'dot':
                sim = self.dot_product(qvec, dvec)
            else:
                sim = self.cosine_similarity(qvec, dvec)
            
            similarities.append((doc_id, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# Usage
docs = ["information retrieval system", "machine learning data", "web search engine"]
vsm = VectorSpaceModel(docs)

# Different similarity measures
print("Cosine:", vsm.search("information system", 'cosine'))
print("Jaccard:", vsm.search("information system", 'jaccard'))
print("Dice:", vsm.search("information system", 'dice'))
print("Dot Product:", vsm.search("information system", 'dot'))
