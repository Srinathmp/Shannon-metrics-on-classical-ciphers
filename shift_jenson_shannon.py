import math, random
from collections import Counter
import numpy as np

# -----------------------
# Helpers: entropy, MI, JSD
# -----------------------
_eps = 1e-12

def _entropy_from_counts(counts, base=2):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    H = 0.0
    for v in counts.values():
        p = v/total
        if p>0:
            H -= p * math.log(p, base)
    return H

def _kl_divergence(p_counts, q_counts, base=2):
    # p_counts, q_counts are Counters over same symbol space (missing keys -> 0)
    all_keys = set(p_counts.keys()) | set(q_counts.keys())
    Np = sum(p_counts.values()) or 1
    Nq = sum(q_counts.values()) or 1
    kl = 0.0
    for k in all_keys:
        p = p_counts.get(k, 0) / Np
        q = q_counts.get(k, 0) / Nq
        if p > 0:
            kl += p * math.log((p + _eps) / (q + _eps), base)
    return kl

def jensen_shannon_divergence(p_counts, q_counts, base=2):
    # JSD = 0.5*KL(p||m) + 0.5*KL(q||m); bounded [0, 1] for base=2
    all_keys = set(p_counts.keys()) | set(q_counts.keys())
    Np = sum(p_counts.values()) or 1
    Nq = sum(q_counts.values()) or 1
    m_counts = Counter()
    for k in all_keys:
        m_counts[k] = (p_counts.get(k,0)/Np + q_counts.get(k,0)/Nq) / 2.0
    kl_pm = 0.0
    kl_qm = 0.0
    for k in all_keys:
        p = p_counts.get(k,0)/Np
        q = q_counts.get(k,0)/Nq
        m = m_counts[k]
        if p>0:
            kl_pm += p * math.log((p + _eps) / (m + _eps), base)
        if q>0:
            kl_qm += q * math.log((q + _eps) / (m + _eps), base)
    return 0.5 * (kl_pm + kl_qm)

def mutual_information_pairs(pairs, base=2):
    # pairs: list of (x,y) observed aligned tokens
    N = len(pairs)
    if N == 0:
        return 0.0
    joint = Counter(pairs)
    x_counts = Counter(x for x,_ in pairs)
    y_counts = Counter(y for _,y in pairs)
    mi = 0.0
    for (x,y), n in joint.items():
        p_xy = n / N
        p_x = x_counts[x] / N
        p_y = y_counts[y] / N
        mi += p_xy * math.log((p_xy + _eps) / (p_x * p_y + _eps), base)
    return mi

# -----------------------
# Improved Confusion (C)
# -----------------------
def confusion_jsd_avg(cipher_func, plaintext, keyspace, sample_pairs=200, base=2):
    """
    Estimate confusion as average Jensen-Shannon divergence between ciphertext distributions
    produced by pairs of keys sampled from keyspace.
    - cipher_func(plaintext, key) -> ciphertext string
    - keyspace: list of keys (or objects the cipher_func accepts)
    Returns C in [0,1] (higher = better confusion)
    """
    if len(keyspace) < 2:
        return 0.0
    # sample pairs
    total_jsd = 0.0
    pairs_done = 0
    for _ in range(sample_pairs):
        k1, k2 = random.sample(keyspace, 2)
        c1 = cipher_func(plaintext, k1)
        c2 = cipher_func(plaintext, k2)
        # counts per symbol
        p1 = Counter(c1)
        p2 = Counter(c2)
        jsd = jensen_shannon_divergence(p1, p2, base=base)
        total_jsd += jsd
        pairs_done += 1
    avg_jsd = total_jsd / pairs_done if pairs_done>0 else 0.0
    # JSD for base=2 is ≤1, so avg_jsd already in [0,1]
    return max(0.0, min(1.0, avg_jsd))

# -----------------------
# Improved Diffusion (D)
# -----------------------
def diffusion_via_bigram_leakage(cipher_func, plaintext, key, bigram=True, base=2):
    """
    Diffusion measured as 1 - normalized_MI(plaintext_grams ; ciphertext_symbols)
    - plaintext: string (we'll extract grams on filtered plaintext)
    - cipher_func(plaintext_variant, key) -> ciphertext string
    - bigram: use plaintext bigrams (True) or unigrams (False)
    Returns D in [0,1], higher => better diffusion (less leakage)
    """
    # prepare filtered plaintext (only letters/digits)
    plain_filtered = "".join(ch for ch in plaintext.upper() if ch.isalnum())
    if bigram:
        grams = [plain_filtered[i:i+2] for i in range(len(plain_filtered)-1)]
    else:
        grams = list(plain_filtered)
    if len(grams) == 0:
        return 0.0
    # produce ciphertext under key
    C = cipher_func(plain_filtered, key)
    # align: we will form pairs (gram_i, C_symbol_i) for i up to min(len(grams), len(C))
    L = min(len(grams), len(C))
    pairs = [(grams[i], C[i]) for i in range(L)]
    mi = mutual_information_pairs(pairs, base=base)
    # entropy of plaintext grams
    H_grams = _entropy_from_counts(Counter(grams), base=base)
    if H_grams <= 0:
        return 0.0
    normalized_leak = mi / H_grams  # in [0,1]
    D = 1.0 - normalized_leak
    return max(0.0, min(1.0, D))

# -----------------------
# Combined linear CDCI
# -----------------------
def cdci_linear(alpha, C, D):
    """
    Linear combination CDCI = alpha * C + (1-alpha) * D
    alpha in [0,1]
    """
    return alpha * C + (1.0 - alpha) * D

# -----------------------
# Convenience wrapper: compute for one cipher/keyset
# -----------------------
def compute_cdci_for_cipher(cipher_func, plaintexts, fixed_key, keyspace,
                            alpha=0.5, sample_pairs=200, bigram=True):
    """
    cipher_func(plaintext, key) -> ciphertext
    plaintexts: list of plaintext strings to average metrics over
    fixed_key: the key to evaluate D on (and for MI bootstrap)
    keyspace: list of keys for confusion sampling
    Returns dictionary with C, D, CDCI and component breakdowns (averaged over plaintexts)
    """
    C_vals = []
    D_vals = []
    for pt in plaintexts:
        # confusion estimated using the same plaintext (structure matters) — average across pts
        C_val = confusion_jsd_avg(lambda p,k: cipher_func(p, k), pt, keyspace, sample_pairs=sample_pairs)
        D_val = diffusion_via_bigram_leakage(lambda p,k: cipher_func(p, k), pt, fixed_key, bigram=bigram)
        C_vals.append(C_val)
        D_vals.append(D_val)
    C_mean = float(np.mean(C_vals))
    D_mean = float(np.mean(D_vals))
    CDCI = cdci_linear(alpha, C_mean, D_mean)
    return {
        'C': C_mean,
        'D': D_mean,
        'CDCI': CDCI,
        'C_per_plain': C_vals,
        'D_per_plain': D_vals
    }

# -----------------------
# Example: Shift (Caesar) usage
# -----------------------
def shift_cipher_wrapper(plaintext, key):
    # key is integer shift 0..25
    text = "".join(ch for ch in plaintext.upper() if 'A' <= ch <= 'Z')
    return "".join(chr(((ord(c)-65+key)%26)+65) for c in text)

if __name__ == "__main__":
    # demo plaintexts
    plaintexts = [("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG"*6)[:200],
                  "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(200)),
                  "HELLOHELLOHELLOHELLOHELLOHELLOHELLOHELLOHELLOHELLO"[:200]]
    fixed_key = 3
    keyspace = list(range(26))  # all Caesar shifts
    out = compute_cdci_for_cipher(shift_cipher_wrapper, plaintexts, fixed_key, keyspace,alpha=0.5, sample_pairs=200, bigram=True)

    print("Shift demo:", out)