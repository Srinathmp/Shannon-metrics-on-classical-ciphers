import math, random
from collections import Counter
import numpy as np

# =======================================================
# Helper: mutual information
# =======================================================
def mutual_information(pairs):
    """Compute empirical mutual information between two variables given as pairs."""
    N = len(pairs)
    joint = Counter(pairs)
    X_counts = Counter(x for x,_ in pairs)
    Y_counts = Counter(y for _,y in pairs)
    mi = 0.0
    for (x,y), n in joint.items():
        p_xy = n/N
        p_x = X_counts[x]/N
        p_y = Y_counts[y]/N
        mi += p_xy * math.log2(p_xy/(p_x*p_y))
    return mi

# =======================================================
# Confusion Component (C-index)
# =======================================================
def confusion_index(cipher_func, plaintext, keyspace, samples=200):
    """
    Estimates confusion: how independent ciphertext is from key.
    Args:
        cipher_func: function (plaintext, key) -> ciphertext string
        plaintext: fixed plaintext string
        keyspace: list of possible keys (subset, not full!)
        samples: number of (key, ciphertext) pairs to use
    Returns:
        C-index in [0,1]
    """
    pairs = []
    for _ in range(samples):
        k = random.choice(keyspace)
        c = cipher_func(plaintext, k)
        # take ciphertext symbols, align them with key label
        for sym in c:
            pairs.append((str(k), sym))
    mi = mutual_information(pairs)
    # normalize by key entropy (log2|keyspace|)
    H_K = math.log2(len(keyspace))
    C = 1 - (mi / H_K if H_K > 0 else 0)
    return max(0, min(1, C))

# =======================================================
# Diffusion Component (D-index)
# =======================================================
def fraction_changed(a, b):
    return sum(x!=y for x,y in zip(a,b)) / len(a)

def symbol_diffusion_index(cipher_func, plaintext, key, alphabet, trials=10):
    """Compute average SDI for random single-symbol flips."""
    L = len(plaintext)
    C = cipher_func(plaintext, key)
    results = []
    for _ in range(trials):
        i = random.randrange(L)
        new_char = random.choice([c for c in alphabet if c != plaintext[i]])
        P2 = plaintext[:i]+new_char+plaintext[i+1:]
        C2 = cipher_func(P2, key)
        results.append(fraction_changed(C, C2))
    return np.mean(results)

def ciphertext_entropy(ciphertexts, alphabet):
    text = "".join(ciphertexts)
    N = len(text)
    counts = Counter(text)
    return -sum((counts[a]/N)*math.log2(counts[a]/N) for a in counts)

def diffusion_index(cipher_func, plaintext, key, alphabet, alpha=0.7, trials=10):
    """
    Weighted combo of SDI and normalized ciphertext entropy.
    """
    sdi = symbol_diffusion_index(cipher_func, plaintext, key, alphabet, trials)
    C = cipher_func(plaintext, key)
    Hc = ciphertext_entropy([C], alphabet)
    norm_entropy = Hc / math.log2(len(alphabet))
    D = alpha*sdi + (1-alpha)*norm_entropy
    return D

# =======================================================
# Combined Confusion–Diffusion Coupled Index (CDCI)
# =======================================================
def cdci_score(cipher_func, plaintext, key, keyspace, alphabet, alpha=0.7):
    C = confusion_index(cipher_func, plaintext, keyspace)
    D = diffusion_index(cipher_func, plaintext, key, alphabet, alpha=alpha)
    return math.sqrt(C*D), C, D
