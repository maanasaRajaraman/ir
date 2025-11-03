import numpy as np

docs = [
    "The football team won the match",
    "He scored a goal in the championship",
    "New AI technology is transforming industry",
    "Machine learning enables smart systems"
]

labels = np.array(["sports", "sports", "tech", "tech"])

test_docs = [
    "The player scored two goals",
    "AI improves computer systems"
]
 
vocab = sorted(set(" ".join(docs + test_docs).lower().split()))

def tf(docs, vocab): 
    return np.array([[doc.lower().split().count(w) for w in vocab] for doc in docs])

def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return (np.dot(a, b) / denom) if denom != 0 else 0.0

def rocchio_train(X, y):
    centroids = {}
    for c in np.unique(y):
        centroids[c] = X[y == c].mean(axis=0)
    return centroids

def rocchio_predict(X, centroids):
    preds = []
    for x in X: 
        sims = {c: cosine_similarity(x, centroid) for c, centroid in centroids.items()}
        preds.append(max(sims, key=sims.get))
    return preds
 
tf_docs = tf(docs, vocab)
tf_test = tf(test_docs, vocab)

rocchio_centroids = rocchio_train(tf_docs, labels)
rocchio_preds = rocchio_predict(tf_test, rocchio_centroids)
for doc, pred in zip(test_docs, rocchio_preds):
    print(f"[{pred}] {doc}")

def train_multinomial_nb(X, y, alpha=1.0):
    classes = np.unique(y)
    n_features = X.shape[1]
    class_priors = {}
    cond_probs = {}
    for c in classes:
        X_c = X[y == c]
        class_priors[c] = X_c.shape[0] / X.shape[0]
        word_counts = X_c.sum(axis=0)
        cond_probs[c] = (word_counts + alpha) / (word_counts.sum() + alpha * n_features)
    return class_priors, cond_probs

def predict_multinomial_nb(X, class_priors, cond_probs):
    preds = []
    for x in X:
        scores = {}
        for c in class_priors:
            log_prob = np.log(class_priors[c]) + np.sum(x * np.log(cond_probs[c]))
            scores[c] = log_prob
        preds.append(max(scores, key=scores.get))
    return preds

print("\n=== MULTINOMIAL NAIVE BAYES ===")
priors_m, cond_probs_m = train_multinomial_nb(tf_docs, labels)
preds_m = predict_multinomial_nb(tf_test, priors_m, cond_probs_m)
for doc, pred in zip(test_docs, preds_m):
    print(f"[{pred}] {doc}")


def train_bernoulli_nb(X, y, alpha=1.0):
    classes = np.unique(y)
    n_features = X.shape[1]
    class_priors = {}
    cond_probs = {}
    for c in classes:
        X_c = X[y == c]
        class_priors[c] = X_c.shape[0] / X.shape[0]
        word_presence = X_c.sum(axis=0)
        cond_probs[c] = (word_presence + alpha) / (X_c.shape[0] + 2 * alpha)
    return class_priors, cond_probs

def predict_bernoulli_nb(X, class_priors, cond_probs):
    preds = []
    for x in X:
        scores = {}
        for c in class_priors:
            p = cond_probs[c]
            log_prob = np.log(class_priors[c]) + np.sum(x * np.log(p) + (1 - x) * np.log(1 - p))
            scores[c] = log_prob
        preds.append(max(scores, key=scores.get))
    return preds

# Manual binarization (replace Binarizer)
X_bin = (tf_docs > 0).astype(int)
X_test_bin = (tf_test > 0).astype(int)

print("\n=== BERNOULLI NAIVE BAYES ===")
priors_b, cond_probs_b = train_bernoulli_nb(X_bin, labels)
preds_b = predict_bernoulli_nb(X_test_bin, priors_b, cond_probs_b)
for doc, pred in zip(test_docs, preds_b):
    print(f"[{pred}] {doc}")
