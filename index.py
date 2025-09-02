class InvertedIndex:
    def __init__(self, docs):
        self.docs = docs
        self.index = {}
        self._build()
    
    def _build(self):
        """Build inverted index"""
        for i, doc in enumerate(self.docs):
            for term in set(doc.lower().split()):
                if term not in self.index:
                    self.index[term] = []
                self.index[term].append(i)
    
    def get(self, term):
        """Get posting list"""
        return self.index.get(term.lower(), [])
    
    def AND(self, list1, list2):
        """Intersect two sorted lists"""
        return [x for x in list1 if x in list2]
    
    def OR(self, list1, list2):
        """Union two lists"""
        return sorted(set(list1 + list2))
    
    def NOT(self, posting_list):
        """Complement of posting list"""
        all_docs = list(range(len(self.docs)))
        return [x for x in all_docs if x not in posting_list]
    
    def optimize_terms(self, terms, operation='and'):
        """Sort terms by posting list length for optimal processing"""
        term_lengths = [(term, len(self.get(term))) for term in terms]
        
        if operation == 'and':
            # For AND: process shortest lists first (fewer intersections)
            return [term for term, _ in sorted(term_lengths, key=lambda x: x[1])]
        else:  # OR
            # For OR: process longest lists first (build result faster)
            return [term for term, _ in sorted(term_lengths, key=lambda x: x[1], reverse=True)]
    
    def search(self, query):
        """Optimized boolean search"""
        q = query.lower()
        
        if ' and ' in q:
            terms = [t.strip() for t in q.split(' and ')]
            # Optimize: shortest posting lists first
            terms = self.optimize_terms(terms, 'and')
            result = self.get(terms[0])
            for term in terms[1:]:
                result = self.AND(result, self.get(term))
                if not result:  # Early termination
                    break
            return result
        
        elif ' or ' in q:
            terms = [t.strip() for t in q.split(' or ')]
            # Optimize: longest posting lists first
            terms = self.optimize_terms(terms, 'or')
            result = self.get(terms[0])
            for term in terms[1:]:
                result = self.OR(result, self.get(term))
            return result
        
        elif ' not ' in q:
            pos, neg = q.split(' not ')
            pos_list = self.get(pos.strip())
            neg_list = self.get(neg.strip())
            return self.AND(pos_list, self.NOT(neg_list))
        
        else:
            return self.get(q)

# Usage with optimization demo
docs = ["cat dog bird", "dog bird", "cat mouse", "bird eagle", "mouse cat"]
idx = InvertedIndex(docs)
print("Index:", idx.index)

# Show optimization in action
print("\nQuery: 'cat and bird and dog'")
terms = ['cat', 'bird', 'dog']
print("Posting list sizes:")
for term in terms:
    print(f"  {term}: {len(idx.get(term))} docs")

optimized = idx.optimize_terms(terms, 'and')
print(f"Optimized order: {optimized}")  # Shortest first
print(f"Result: {idx.search('cat and bird and dog')}")

print(f"\nOR optimization:")
or_optimized = idx.optimize_terms(terms, 'or') 
print(f"OR order: {or_optimized}")  # Longest first
