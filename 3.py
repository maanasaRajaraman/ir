# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 11:53:28 2025

@author: preth
"""

class TermDocumentBooleanModel:
    def __init__(self, documents):
        self.documents = documents
        self.vocab = []
        self.term_doc_matrix = []
        self._build_matrix()
    
    def _build_matrix(self):
        """Build term-document matrix"""
        # Get all unique terms
        all_terms = set()
        processed_docs = []
        
        for doc in self.documents:
            tokens = doc.lower().split()
            processed_docs.append(tokens)
            all_terms.update(tokens)
        
        self.vocab = sorted(all_terms)
        
        # Build matrix: rows=terms, cols=documents
        self.term_doc_matrix = []
        for term in self.vocab:
            row = []
            for doc_tokens in processed_docs:
                row.append(1 if term in doc_tokens else 0)
            self.term_doc_matrix.append(row)
    
    def get_term_vector(self, term):
        """Get document vector for a term"""
        term = term.lower()
        if term not in self.vocab:
            return [0] * len(self.documents)
        
        term_idx = self.vocab.index(term)
        return self.term_doc_matrix[term_idx]
    
    def boolean_and(self, term1, term2):
        """Boolean AND operation"""
        vec1 = self.get_term_vector(term1)
        vec2 = self.get_term_vector(term2)
        return [a & b for a, b in zip(vec1, vec2)]
    
    def boolean_or(self, term1, term2):
        """Boolean OR operation"""
        vec1 = self.get_term_vector(term1)
        vec2 = self.get_term_vector(term2)
        return [a | b for a, b in zip(vec1, vec2)]
    
    def boolean_not(self, term):
        """Boolean NOT operation"""
        vec = self.get_term_vector(term)
        return [1 - x for x in vec]
    
    def search(self, query):
        """Search with boolean operators"""
        query = query.lower().strip()
        
        # Single term
        if ' ' not in query:
            result_vector = self.get_term_vector(query)
        
        # AND operation
        elif ' and ' in query:
            terms = [t.strip() for t in query.split(' and ')]
            result_vector = self.get_term_vector(terms[0])
            for term in terms[1:]:
                term_vec = self.get_term_vector(term)
                result_vector = [a & b for a, b in zip(result_vector, term_vec)]
        
        # OR operation
        elif ' or ' in query:
            terms = [t.strip() for t in query.split(' or ')]
            result_vector = self.get_term_vector(terms[0])
            for term in terms[1:]:
                term_vec = self.get_term_vector(term)
                result_vector = [a | b for a, b in zip(result_vector, term_vec)]
        
        # NOT operation
        elif ' not ' in query:
            parts = query.split(' not ')
            pos_term = parts[0].strip()
            neg_term = parts[1].strip()
            
            pos_vec = self.get_term_vector(pos_term)
            neg_vec = self.get_term_vector(neg_term)
            neg_vec = [1 - x for x in neg_vec]  # NOT operation
            result_vector = [a & b for a, b in zip(pos_vec, neg_vec)]
        
        else:
            result_vector = [0] * len(self.documents)
        
        # Return document IDs where result is 1
        return [i for i, val in enumerate(result_vector) if val == 1]
    
    def print_matrix(self):
        """Print term-document matrix"""
        print("Term-Document Matrix:")
        print("Terms\\Docs", end="")
        for i in range(len(self.documents)):
            print(f"\tD{i}", end="")
        print()
        
        for i, term in enumerate(self.vocab):
            print(f"{term:<10}", end="")
            for val in self.term_doc_matrix[i]:
                print(f"\t{val}", end="")
            print()

# Usage Example
if __name__ == "__main__":
    # Sample documents
    docs = [
        "information retrieval system",
        "database search query",
        "information system database",
        "web search engine",
        "query processing system"
    ]
    
    model = TermDocumentBooleanModel(docs)
    model.print_matrix()
    
    print("\nSearch Results:")
    print("'information':", model.search("information"))
    print("'information and system':", model.search("information and system"))
    print("'search or query':", model.search("search or query"))
    print("'system not database':", model.search("system not database"))
