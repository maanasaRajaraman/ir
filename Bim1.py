# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 12:24:17 2025

@author: preth
"""

import math

class BinaryIndependenceModel:
    def __init__(self, docs):
        self.docs = docs
        self.vocab, self.binary_matrix = self._preprocess()
        self.N_d = len(self.binary_matrix)  # Total documents
    
    def _preprocess(self):
        """Integrated preprocessing for BIM"""
        vocab_set = set()
        processed = []
        
        for doc in self.docs:
            tokens = set(doc.lower().split())  # Unique tokens per doc
            processed.append(tokens)
            vocab_set.update(tokens)
        
        vocab = sorted(vocab_set)
        
        # Binary matrix
        binary_matrix = []
        for doc_tokens in processed:
            row = [1 if term in doc_tokens else 0 for term in vocab]
            binary_matrix.append(row)
        
        return vocab, binary_matrix
    
    def phase1_estimate(self, query_terms):
        """Phase I: Estimation without relevance information"""
        estimates = {}
        
        for term in query_terms:
            if term not in self.vocab:
                continue
            
            term_idx = self.vocab.index(term)
            
            # d_k = document frequency (number of docs containing term k)
            d_k = sum(1 for doc in self.binary_matrix if doc[term_idx] == 1)
            
            # Phase I formulas (without smoothing first, then with)
            # p_k ≈ 0.5 (assume random chance)
            p_k = 0.5
            
            # q_k ≈ d_k / N_d (simple estimation)
            q_k_simple = d_k / self.N_d if self.N_d > 0 else 0
            
            # q_k with smoothing = (d_k + 0.5) / (N_d + 1)
            q_k = (d_k + 0.5) / (self.N_d + 1)
            
            estimates[term] = {
                'd_k': d_k,
                'p_k': p_k,
                'q_k_simple': q_k_simple,
                'q_k': q_k
            }
        
        return estimates
    
    def phase2_estimate(self, query_terms, relevant_docs):
        """Phase II: Estimation with relevance information"""
        estimates = {}
        N_r = len(relevant_docs)  # Number of relevant docs
        
        for term in query_terms:
            if term not in self.vocab:
                continue
            
            term_idx = self.vocab.index(term)
            
            # r_k = number of relevant docs containing term k
            r_k = sum(1 for doc_id in relevant_docs 
                     if self.binary_matrix[doc_id][term_idx] == 1)
            
            # d_k = total docs containing term k  
            d_k = sum(1 for doc in self.binary_matrix if doc[term_idx] == 1)
            
            # Phase II formulas
            # p_k = r_k / N_r (without smoothing)
            p_k_simple = r_k / N_r if N_r > 0 else 0.5
            
            # p_k = (r_k + 0.5) / (N_r + 1) (with smoothing)
            p_k = (r_k + 0.5) / (N_r + 1)
            
            # q_k = (d_k - r_k) / (N_d - N_r) (without smoothing)
            q_k_simple = (d_k - r_k) / (self.N_d - N_r) if (self.N_d - N_r) > 0 else 0.5
            
            # q_k = (d_k - r_k + 0.5) / (N_d - N_r + 1) (with smoothing)
            q_k = (d_k - r_k + 0.5) / (self.N_d - N_r + 1)
            
            estimates[term] = {
                'r_k': r_k,
                'd_k': d_k,
                'N_r': N_r,
                'p_k_simple': p_k_simple,
                'p_k': p_k,
                'q_k_simple': q_k_simple,
                'q_k': q_k
            }
        
        return estimates
    
    def calculate_rsv(self, doc_id, query_terms, estimates):
        """Calculate Retrieval Status Value"""
        rsv = 0
        
        for term in query_terms:
            if term not in estimates:
                continue
            
            term_idx = self.vocab.index(term)
            p_k = estimates[term]['p_k']
            q_k = estimates[term]['q_k']
            
            # Check if document contains term
            if self.binary_matrix[doc_id][term_idx] == 1:
                # Document contains term: log(p_k / q_k)
                if p_k > 0 and q_k > 0:
                    rsv += math.log(p_k / q_k)
            else:
                # Document doesn't contain term: log((1-p_k) / (1-q_k))
                if p_k < 1 and q_k < 1:
                    rsv += math.log((1 - p_k) / (1 - q_k))
        
        return rsv
    
    def search_phase1(self, query, top_k=5):
        """Phase I search without relevance feedback"""
        query_terms = query.lower().split()
        estimates = self.phase1_estimate(query_terms)
        
        doc_scores = []
        for doc_id in range(self.N_d):
            rsv = self.calculate_rsv(doc_id, query_terms, estimates)
            doc_scores.append((doc_id, rsv))
        
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:top_k]
    
    def search_phase2(self, query, relevant_docs, top_k=5):
        """Phase II search with relevance feedback"""
        query_terms = query.lower().split()
        estimates = self.phase2_estimate(query_terms, relevant_docs)
        
        doc_scores = []
        for doc_id in range(self.N_d):
            rsv = self.calculate_rsv(doc_id, query_terms, estimates)
            doc_scores.append((doc_id, rsv))
        
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:top_k]
    
    def print_estimates(self, query_terms, relevant_docs=None):
        """Print probability estimates for analysis"""
        if relevant_docs is None:
            print("=== PHASE I ESTIMATES ===")
            estimates = self.phase1_estimate(query_terms)
            for term, est in estimates.items():
                print(f"{term}: d_k={est['d_k']}, p_k={est['p_k']:.3f}, q_k={est['q_k']:.3f}")
        else:
            print("=== PHASE II ESTIMATES ===")
            estimates = self.phase2_estimate(query_terms, relevant_docs)
            for term, est in estimates.items():
                print(f"{term}: r_k={est['r_k']}, d_k={est['d_k']}, N_r={est['N_r']}")
                print(f"      p_k={est['p_k']:.3f}, q_k={est['q_k']:.3f}")

# Usage Example
if __name__ == "__main__":
    docs = [
        "information retrieval system",
        "database search query", 
        "information system database",
        "web search engine",
        "query processing system"
    ]
    
    bim = BinaryIndependenceModel(docs)
    
    query = "information system"
    query_terms = query.split()
    
    print("=== PHASE I (No Relevance Info) ===")
    bim.print_estimates(query_terms)
    results1 = bim.search_phase1(query)
    print(f"Phase I Results: {results1}")
    
    print("\n=== PHASE II (With Relevance Feedback) ===")
    relevant_docs = [0, 2]  # Assume docs 0,2 are relevant
    bim.print_estimates(query_terms, relevant_docs)
    results2 = bim.search_phase2(query, relevant_docs)
    print(f"Phase II Results: {results2}")

# Quick memory formulas
def bim_formulas():
    """Key BIM formulas to remember"""
    print("Phase I (No relevance info):")
    print("  p_k = 0.5")
    print("  q_k = (d_k + 0.5) / (N_d + 1)")
    
    print("\nPhase II (With relevance info):")
    print("  p_k = (r_k + 0.5) / (N_r + 1)")
    print("  q_k = (d_k - r_k + 0.5) / (N_d - N_r + 1)")
    
    print("\nRSV calculation:")
    print("  If term in doc: RSV += log(p_k / q_k)")
    print("  If term not in doc: RSV += log((1-p_k) / (1-q_k))")