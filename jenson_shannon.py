# all_ciphers_cdci.py
# Single-file evaluation for 6 classical ciphers with improved CDCI (JSD confusion + bigram-MI diffusion)
# Produces per-cipher fixed/random DataFrames, plots, and CSVs.
#
# Requirements: numpy, pandas, matplotlib, sympy (for Hill). caas_jupyter_tools.#display_dataframe_to_user is used for DataFrames.
# Save as a file and run, or run in a Jupyter environment.

import random, math, numpy as np, pandas as pd, matplotlib.pyplot as plt
from collections import Counter
# from caas_jupyter_tools import #display_dataframe_to_user

# If sympy is needed for Hill:
try:
    from sympy import Matrix
except Exception:
    Matrix = None

# ----------------------------
# Shared metric functions
# ----------------------------
_eps = 1e-12
def _entropy_from_counts(counts, base=2):
    total = sum(counts.values())
    if total == 0: return 0.0
    H = 0.0
    for v in counts.values():
        p = v/total
        if p>0:
            H -= p * math.log(p, base)
    return H

def jensen_shannon_divergence(p_counts, q_counts, base=2):
    all_keys = set(p_counts.keys()) | set(q_counts.keys())
    Np, Nq = sum(p_counts.values()) or 1, sum(q_counts.values()) or 1
    m = {}
    for k in all_keys:
        m[k] = (p_counts.get(k,0)/Np + q_counts.get(k,0)/Nq) / 2.0
    kl_pm = 0.0; kl_qm = 0.0
    for k in all_keys:
        p = p_counts.get(k,0)/Np; q = q_counts.get(k,0)/Nq; mm = m[k]
        if p>0: kl_pm += p * math.log((p+_eps)/(mm+_eps), base)
        if q>0: kl_qm += q * math.log((q+_eps)/(mm+_eps), base)
    return 0.5*(kl_pm+kl_qm)

def mutual_information_pairs(pairs, base=2):
    N = len(pairs)
    if N==0: return 0.0
    joint = Counter(pairs)
    X = Counter(x for x,_ in pairs); Y = Counter(y for _,y in pairs)
    mi = 0.0
    for (x,y),n in joint.items():
        p_xy = n/N; p_x = X[x]/N; p_y = Y[y]/N
        mi += p_xy * math.log((p_xy+_eps)/(p_x*p_y+_eps), base)
    return mi

def confusion_jsd_avg(cipher_fun, plaintext, keyspace, sample_pairs=200, base=2):
    if len(keyspace) < 2: return 0.0
    total = 0.0; count = 0
    # If keyspace small, enumerate pairs; else sample
    for _ in range(sample_pairs):
        k1, k2 = random.sample(keyspace, 2)
        c1 = cipher_fun(plaintext, k1)
        c2 = cipher_fun(plaintext, k2)
        total += jensen_shannon_divergence(Counter(c1), Counter(c2), base=base)
        count += 1
    return total/count if count else 0.0

def diffusion_via_bigram_leakage(cipher_fun, plaintext, key, base=2):
    pt = "".join(ch for ch in plaintext.upper() if ch.isalnum())
    grams = [pt[i:i+2] for i in range(len(pt)-1)]
    if not grams:
        return 0.0
    C = cipher_fun(pt, key)
    L = min(len(grams), len(C))
    pairs = [(grams[i], C[i]) for i in range(L)]
    mi = mutual_information_pairs(pairs, base=base)
    H = _entropy_from_counts(Counter(grams), base=base)
    return 1.0 - (mi / H if H > 0 else 0.0)

def cdci_linear(alpha, C, D): return alpha*C + (1.0-alpha)*D

def compute_cdci_for_cipher(cipher_func, plaintexts, fixed_key, keyspace, alpha=0.5, sample_pairs=200):
    Cs = []; Ds = []
    for pt in plaintexts:
        C_val = confusion_jsd_avg(cipher_func, pt, keyspace, sample_pairs=sample_pairs)
        D_val = diffusion_via_bigram_leakage(cipher_func, pt, fixed_key)
        Cs.append(C_val); Ds.append(D_val)
    C_mean = float(np.mean(Cs)); D_mean = float(np.mean(Ds))
    return {'C':C_mean, 'D':D_mean, 'CDCI': cdci_linear(alpha, C_mean, D_mean), 'C_per_plain': Cs, 'D_per_plain': Ds}

# Generic helper metrics reused in each block
def fraction_changed(a, b):
    L = min(len(a), len(b))
    return sum(x!=y for x,y in zip(a[:L], b[:L])) / L if L else 0.0

def symbol_diffusion_index(cipher_func, plaintext, key, alphabet, trials=20):
    base_ct = cipher_func(plaintext, key)
    results = []
    pt_filtered = "".join(ch for ch in plaintext.upper() if ch.isalnum())
    if len(pt_filtered) == 0:
        return 0.0, 0.0
    for _ in range(trials):
        i = random.randrange(len(pt_filtered))
        new_char = random.choice([c for c in alphabet if c != pt_filtered[i]])
        P2 = pt_filtered[:i] + new_char + pt_filtered[i+1:]
        C2 = cipher_func(P2, key)
        results.append(fraction_changed(base_ct, C2))
    return float(np.mean(results)), float(np.std(results)) if len(results)>1 else 0.0

def compute_ksi(cipher_func, plaintext, key, key_variant):
    C1 = cipher_func(plaintext, key)
    C2 = cipher_func(plaintext, key_variant)
    return fraction_changed(C1, C2)

def compute_mi_from_plain_cipher_lists(plain_list, cipher_list):
    conc_plain = "".join( "".join(ch for ch in p.upper() if ch.isalnum()) for p in plain_list )
    conc_cipher = "".join(cipher_list)
    L = min(len(conc_plain), len(conc_cipher))
    pairs = list(zip(conc_plain[:L], conc_cipher[:L]))
    return mutual_information_pairs(pairs)

def compute_chi_entropy(ciphertexts, alphabet):
    text = "".join(ciphertexts)
    N = len(text)
    if N == 0: return 0.0, 0.0
    counts = Counter(text)
    expected = N / len(alphabet)
    chi2 = sum((counts[a] - expected)**2 / expected for a in alphabet)
    entropy = -sum((counts[a]/N) * math.log2(counts[a]/N) for a in counts if counts[a]>0)
    return chi2, entropy

def bootstrap_mi(pairs, n_boot=300, alpha=0.05):
    N = len(pairs)
    if N == 0: return 0.0, (0.0, 0.0)
    estimates = []
    for _ in range(n_boot):
        sample = [pairs[random.randrange(N)] for _ in range(N)]
        estimates.append(mutual_information_pairs(sample))
    estimates.sort()
    low = estimates[int((alpha/2)*n_boot)]
    high = estimates[int((1-alpha/2)*n_boot)]
    return np.mean(estimates), (low, high)

# ----------------------------
# 1) Shift (Caesar)
# ----------------------------
def shift_encrypt(text, key):
    text = "".join(ch for ch in text.upper() if 'A' <= ch <= 'Z')
    return "".join(chr(((ord(c)-65 + key) % 26) + 65) for c in text)

def evaluate_shift(num_texts=5, length=200, fixed_key=3, random_key_count=10, alpha=0.5):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    plaintexts = [ "".join(random.choice(alphabet) for _ in range(length)) for _ in range(num_texts) ]
    plaintexts[0] = ("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG" * 6)[:length]

    keyspace = list(range(26))
    # Fixed mode
    fixed_metrics = {}
    sdi_vals=[]; sdi_stds=[]; ksi_vals=[]; mi_vals=[]; chi_vals=[]; ent_vals=[]; mi_boot_means=[]
    for pt in plaintexts:
        sdi_mean, sdi_std = symbol_diffusion_index(lambda p,k: shift_encrypt(p,k), pt, fixed_key, alphabet)
        sdi_vals.append(sdi_mean); sdi_stds.append(sdi_std)
        ksi_vals.append(compute_ksi(lambda p,k: shift_encrypt(p,k), pt, fixed_key, (fixed_key+1)%26))
        C = shift_encrypt(pt, fixed_key)
        mi_vals.append(compute_mi_from_plain_cipher_lists([pt],[C]))
        chi,H = compute_chi_entropy([C], alphabet); chi_vals.append(chi); ent_vals.append(H)
        pairs = list(zip("".join(ch for ch in pt.upper() if ch.isalnum()), C[:len(pt)]))
        mb,ci = bootstrap_mi(pairs, n_boot=200); mi_boot_means.append(mb)

    cdci_out = compute_cdci_for_cipher(lambda p,k: shift_encrypt(p,k), plaintexts, fixed_key, keyspace, alpha=alpha, sample_pairs=200)

    fixed_metrics.update({
        'SDI_mean': np.mean(sdi_vals), 'SDI_std': np.mean(sdi_stds),
        'KSI_mean': np.mean(ksi_vals), 'MI_mean': np.mean(mi_vals),
        'Chi2_mean': np.mean(chi_vals), 'Entropy_mean': np.mean(ent_vals),
        'MI_boot_mean': np.mean(mi_boot_means),
        'C': cdci_out['C'], 'D': cdci_out['D'], 'CDCI': cdci_out['CDCI']
    })
    fixed_df = pd.DataFrame([fixed_metrics])

    # Random mode
    random_keys = random.sample(range(26), random_key_count)
    rand_records=[]
    for rk in random_keys:
        sdi_vals=[]; ksi_vals=[]; mi_vals=[]; chi_vals=[]; ent_vals=[]; mi_boot=[]
        for pt in plaintexts:
            sm,ss = symbol_diffusion_index(lambda p,k: shift_encrypt(p,k), pt, rk, alphabet)
            sdi_vals.append(sm); ksi_vals.append(compute_ksi(lambda p,k: shift_encrypt(p,k), pt, rk, (rk+1)%26))
            C = shift_encrypt(pt, rk)
            mi_vals.append(compute_mi_from_plain_cipher_lists([pt],[C])); chi,H = compute_chi_entropy([C], alphabet)
            chi_vals.append(chi); ent_vals.append(H)
            mb,_ = bootstrap_mi(list(zip("".join(ch for ch in pt.upper() if ch.isalnum()), C[:len(pt)])), n_boot=120)
            mi_boot.append(mb)
        cdci_out = compute_cdci_for_cipher(lambda p,k: shift_encrypt(p,k), plaintexts, rk, keyspace, alpha=alpha, sample_pairs=120)
        rand_records.append({
            'key': rk, 'SDI_mean': np.mean(sdi_vals), 'KSI_mean': np.mean(ksi_vals),
            'MI_mean': np.mean(mi_vals), 'Chi2_mean': np.mean(chi_vals), 'Entropy_mean': np.mean(ent_vals),
            'MI_boot_mean': np.mean(mi_boot), 'C': cdci_out['C'], 'D': cdci_out['D'], 'CDCI': cdci_out['CDCI']
        })
    random_df = pd.DataFrame(rand_records)
    summary = pd.DataFrame({
        'mode':['fixed','random_mean'],
        'SDI_mean':[fixed_metrics['SDI_mean'], random_df['SDI_mean'].mean()],
        'KSI_mean':[fixed_metrics['KSI_mean'], random_df['KSI_mean'].mean()],
        'MI_mean':[fixed_metrics['MI_mean'], random_df['MI_mean'].mean()],
        'Entropy_mean':[fixed_metrics['Entropy_mean'], random_df['Entropy_mean'].mean()],
        'CDCI':[fixed_metrics['CDCI'], random_df['CDCI'].mean()],
        'C_index':[fixed_metrics['C'], random_df['C'].mean()],
        'D_index':[fixed_metrics['D'], random_df['D'].mean()]
    })

    # #display_dataframe_to_user("Shift - Fixed", fixed_df)
    # #display_dataframe_to_user("Shift - Random", random_df)
    # #display_dataframe_to_user("Shift - Summary", summary)

    # plots
    x = np.arange(len(summary)); width=0.25
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(x-width, summary['CDCI'], width, label='CDCI')
    ax.bar(x, summary['C_index'], width, label='C-index')
    ax.bar(x+width, summary['D_index'], width, label='D-index')
    ax.set_xticks(x); ax.set_xticklabels(summary['mode'])
    ax.set_ylabel('Score'); ax.set_title('Shift: CDCI / C / D'); ax.legend(); plt.show()

    fixed_df.to_csv('./mnt/data/shift_fixed.csv', index=False)
    random_df.to_csv('./mnt/data/shift_random.csv', index=False)
    summary.to_csv('./mnt/data/shift_summary.csv', index=False)
    print("Shift saved to ./mnt/data/shift_*.csv")
    return fixed_df, random_df, summary

# ----------------------------
# 2) Substitution cipher
# ----------------------------
ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
def substitution_encrypt(text, key):
    mapping = {ALPH[i]: key[i] for i in range(26)}
    txt = "".join(ch for ch in text.upper() if 'A' <= ch <= 'Z')
    return "".join(mapping[c] for c in txt)

def random_sub_key():
    letters = list(ALPH); random.shuffle(letters); return "".join(letters)

def perturb_sub_key(key):
    k = list(key)
    i,j = random.sample(range(26),2); k[i],k[j]=k[j],k[i]
    return "".join(k)

def evaluate_substitution(num_texts=5, length=200, fixed_key="QWERTYUIOPASDFGHJKLZXCVBNM", random_key_count=10, alpha=0.5):
    plaintexts = [ "".join(random.choice(ALPH) for _ in range(length)) for _ in range(num_texts) ]
    plaintexts[0] = ("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG" * 6)[:length]
    keyspace = [random_sub_key() for _ in range(120)]  # sample space for confusion JS divergence

    # Fixed
    fixed_metrics = {}
    sdi_vals=[]; ksi_vals=[]; mi_vals=[]; chi_vals=[]; ent_vals=[]; mi_boot=[]
    for pt in plaintexts:
        sm,ss = symbol_diffusion_index(substitution_encrypt, pt, fixed_key, ALPH)
        sdi_vals.append(sm); ksi_vals.append(compute_ksi(substitution_encrypt, pt, fixed_key, perturb_sub_key(fixed_key)))
        C = substitution_encrypt(pt, fixed_key)
        mi_vals.append(compute_mi_from_plain_cipher_lists([pt],[C])); chi,H = compute_chi_entropy([C], ALPH)
        chi_vals.append(chi); ent_vals.append(H)
        mb,ci = bootstrap_mi(list(zip("".join(ch for ch in pt.upper() if ch.isalnum()), C[:len(pt)])), n_boot=150)
        mi_boot.append(mb)
    cdci_out = compute_cdci_for_cipher(substitution_encrypt, plaintexts, fixed_key, keyspace, alpha=alpha)
    fixed_metrics.update({
        'SDI_mean': np.mean(sdi_vals), 'KSI_mean': np.mean(ksi_vals), 'MI_mean': np.mean(mi_vals),
        'Chi2_mean': np.mean(chi_vals), 'Entropy_mean': np.mean(ent_vals), 'MI_boot_mean': np.mean(mi_boot),
        'C': cdci_out['C'], 'D': cdci_out['D'], 'CDCI': cdci_out['CDCI']
    })
    fixed_df = pd.DataFrame([fixed_metrics])

    # Random keys
    random_keys = [random_sub_key() for _ in range(random_key_count)]
    rand_records=[]
    for rk in random_keys:
        sdi_vals=[]; ksi_vals=[]; mi_vals=[]; chi_vals=[]; ent_vals=[]; mi_boot=[]
        for pt in plaintexts:
            sm,ss = symbol_diffusion_index(substitution_encrypt, pt, rk, ALPH)
            sdi_vals.append(sm); ksi_vals.append(compute_ksi(substitution_encrypt, pt, rk, perturb_sub_key(rk)))
            C = substitution_encrypt(pt, rk)
            mi_vals.append(compute_mi_from_plain_cipher_lists([pt],[C])); chi,H = compute_chi_entropy([C], ALPH)
            chi_vals.append(chi); ent_vals.append(H)
            mb,_ = bootstrap_mi(list(zip("".join(ch for ch in pt.upper() if ch.isalnum()), C[:len(pt)])), n_boot=120)
            mi_boot.append(mb)
        cdci_out = compute_cdci_for_cipher(substitution_encrypt, plaintexts, rk, keyspace, alpha=alpha)
        rand_records.append({
            'key': rk, 'SDI_mean': np.mean(sdi_vals), 'KSI_mean': np.mean(ksi_vals),
            'MI_mean': np.mean(mi_vals), 'Chi2_mean': np.mean(chi_vals), 'Entropy_mean': np.mean(ent_vals),
            'MI_boot_mean': np.mean(mi_boot), 'C': cdci_out['C'], 'D': cdci_out['D'], 'CDCI': cdci_out['CDCI']
        })
    random_df = pd.DataFrame(rand_records)
    summary = pd.DataFrame({
        'mode':['fixed','random_mean'],
        'SDI_mean':[fixed_metrics['SDI_mean'], random_df['SDI_mean'].mean()],
        'KSI_mean':[fixed_metrics['KSI_mean'], random_df['KSI_mean'].mean()],
        'MI_mean':[fixed_metrics['MI_mean'], random_df['MI_mean'].mean()],
        'Entropy_mean':[fixed_metrics['Entropy_mean'], random_df['Entropy_mean'].mean()],
        'CDCI':[fixed_metrics['CDCI'], random_df['CDCI'].mean()],
        'C_index':[fixed_metrics['C'], random_df['C'].mean()],
        'D_index':[fixed_metrics['D'], random_df['D'].mean()]
    })

    ##display_dataframe_to_user("Substitution - Fixed", fixed_df)
    #display_dataframe_to_user("Substitution - Random", random_df)
    #display_dataframe_to_user("Substitution - Summary", summary)

    # plots
    x = np.arange(len(summary)); width=0.25
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(x-width, summary['CDCI'], width, label='CDCI')
    ax.bar(x, summary['C_index'], width, label='C-index')
    ax.bar(x+width, summary['D_index'], width, label='D-index')
    ax.set_xticks(x); ax.set_xticklabels(summary['mode']); ax.set_ylabel('Score'); ax.set_title('Substitution: CDCI / C / D'); ax.legend(); plt.show()

    fixed_df.to_csv('./mnt/data/substitution_fixed.csv', index=False)
    random_df.to_csv('./mnt/data/substitution_random.csv', index=False)
    summary.to_csv('./mnt/data/substitution_summary.csv', index=False)
    print("Substitution saved to ./mnt/data/substitution_*.csv")
    return fixed_df, random_df, summary

# ----------------------------
# 3) Vigenere cipher
# ----------------------------
def vigenere_encrypt(text, key):
    txt = "".join(ch for ch in text.upper() if 'A' <= ch <= 'Z')
    key = key.upper()
    out = []
    for i,ch in enumerate(txt):
        shift = ord(key[i % len(key)]) - 65
        out.append(chr(((ord(ch)-65 + shift) % 26) + 65))
    return "".join(out)

def random_vigenere_key(min_len=3, max_len=7):
    return "".join(random.choice(ALPH) for _ in range(random.randint(min_len, max_len)))

def perturb_vigenere_key(key):
    pos = random.randrange(len(key)); new_char = random.choice([c for c in ALPH if c!=key[pos]])
    return key[:pos] + new_char + key[pos+1:]

def evaluate_vigenere(num_texts=5, length=200, fixed_key="KEY", random_key_count=10, alpha=0.5):
    plaintexts = [ "".join(random.choice(ALPH) for _ in range(length)) for _ in range(num_texts) ]
    plaintexts[0] = ("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG" * 6)[:length]
    keyspace = [random_vigenere_key() for _ in range(120)]

    fixed_metrics = {}
    sdi_vals=[]; ksi_vals=[]; mi_vals=[]; chi_vals=[]; ent_vals=[]; mi_boot=[]
    for pt in plaintexts:
        sm,ss = symbol_diffusion_index(vigenere_encrypt, pt, fixed_key, ALPH)
        sdi_vals.append(sm); ksi_vals.append(compute_ksi(vigenere_encrypt, pt, fixed_key, perturb_vigenere_key(fixed_key)))
        C = vigenere_encrypt(pt, fixed_key)
        mi_vals.append(compute_mi_from_plain_cipher_lists([pt],[C])); chi,H = compute_chi_entropy([C], ALPH)
        chi_vals.append(chi); ent_vals.append(H)
        mb,ci = bootstrap_mi(list(zip("".join(ch for ch in pt.upper() if ch.isalnum()), C[:len(pt)])), n_boot=150)
        mi_boot.append(mb)
    cdci_out = compute_cdci_for_cipher(vigenere_encrypt, plaintexts, fixed_key, keyspace, alpha=alpha)
    fixed_metrics.update({
        'SDI_mean': np.mean(sdi_vals), 'KSI_mean': np.mean(ksi_vals), 'MI_mean': np.mean(mi_vals),
        'Chi2_mean': np.mean(chi_vals), 'Entropy_mean': np.mean(ent_vals), 'MI_boot_mean': np.mean(mi_boot),
        'C': cdci_out['C'], 'D': cdci_out['D'], 'CDCI': cdci_out['CDCI']
    })
    fixed_df = pd.DataFrame([fixed_metrics])

    random_keys = [random_vigenere_key() for _ in range(random_key_count)]
    rand_records=[]
    for rk in random_keys:
        sdi_vals=[]; ksi_vals=[]; mi_vals=[]; chi_vals=[]; ent_vals=[]; mi_boot=[]
        for pt in plaintexts:
            sm,ss = symbol_diffusion_index(vigenere_encrypt, pt, rk, ALPH)
            sdi_vals.append(sm); ksi_vals.append(compute_ksi(vigenere_encrypt, pt, rk, perturb_vigenere_key(rk)))
            C = vigenere_encrypt(pt, rk)
            mi_vals.append(compute_mi_from_plain_cipher_lists([pt],[C])); chi,H = compute_chi_entropy([C], ALPH)
            chi_vals.append(chi); ent_vals.append(H)
            mb,_ = bootstrap_mi(list(zip("".join(ch for ch in pt.upper() if ch.isalnum()), C[:len(pt)])), n_boot=120)
            mi_boot.append(mb)
        cdci_out = compute_cdci_for_cipher(vigenere_encrypt, plaintexts, rk, keyspace, alpha=alpha)
        rand_records.append({
            'key': rk, 'SDI_mean': np.mean(sdi_vals), 'KSI_mean': np.mean(ksi_vals),
            'MI_mean': np.mean(mi_vals), 'Chi2_mean': np.mean(chi_vals), 'Entropy_mean': np.mean(ent_vals),
            'MI_boot_mean': np.mean(mi_boot), 'C': cdci_out['C'], 'D': cdci_out['D'], 'CDCI': cdci_out['CDCI']
        })
    random_df = pd.DataFrame(rand_records)
    summary = pd.DataFrame({
        'mode':['fixed','random_mean'],
        'SDI_mean':[fixed_metrics['SDI_mean'], random_df['SDI_mean'].mean()],
        'KSI_mean':[fixed_metrics['KSI_mean'], random_df['KSI_mean'].mean()],
        'MI_mean':[fixed_metrics['MI_mean'], random_df['MI_mean'].mean()],
        'Entropy_mean':[fixed_metrics['Entropy_mean'], random_df['Entropy_mean'].mean()],
        'CDCI':[fixed_metrics['CDCI'], random_df['CDCI'].mean()],
        'C_index':[fixed_metrics['C'], random_df['C'].mean()],
        'D_index':[fixed_metrics['D'], random_df['D'].mean()]
    })

    #display_dataframe_to_user("Vigenere - Fixed", fixed_df)
    #display_dataframe_to_user("Vigenere - Random", random_df)
    #display_dataframe_to_user("Vigenere - Summary", summary)

    # plots
    x = np.arange(len(summary)); width=0.25
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(x-width, summary['CDCI'], width, label='CDCI')
    ax.bar(x, summary['C_index'], width, label='C-index')
    ax.bar(x+width, summary['D_index'], width, label='D-index')
    ax.set_xticks(x); ax.set_xticklabels(summary['mode']); ax.set_ylabel('Score'); ax.set_title('Vigenere: CDCI / C / D'); ax.legend(); plt.show()

    fixed_df.to_csv('./mnt/data/vigenere_fixed.csv', index=False)
    random_df.to_csv('./mnt/data/vigenere_random.csv', index=False)
    summary.to_csv('./mnt/data/vigenere_summary.csv', index=False)
    print("Vigenere saved to ./mnt/data/vigenere_*.csv")
    return fixed_df, random_df, summary

# ----------------------------
# 4) Columnar Transposition
# ----------------------------
def columnar_transposition_encrypt(text, key):
    txt = "".join(ch for ch in text.upper() if ch.isalnum())
    n = len(key)
    rows = (len(txt) + n - 1) // n
    padded = txt + "X"*(rows*n - len(txt))
    matrix = [padded[i*n:(i+1)*n] for i in range(rows)]
    order = sorted(range(n), key=lambda i: key[i])
    ciphertext = "".join("".join(row[j] for row in matrix) for j in order)
    return ciphertext

def random_trans_key(min_len=4, max_len=8):
    length = random.randint(min_len, max_len)
    return "".join(random.choice(ALPH) for _ in range(length))

def perturb_trans_key(key):
    k = list(key)
    if len(k) < 2: return key
    i,j = random.sample(range(len(k)),2); k[i],k[j]=k[j],k[i]
    return "".join(k)

def evaluate_transposition(num_texts=5, length=200, fixed_key="ZEBRAS", random_key_count=10, alpha=0.5):
    plaintexts = [ "".join(random.choice(ALPH) for _ in range(length)) for _ in range(num_texts) ]
    plaintexts[0] = ("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG" * 6)[:length]
    keyspace = [random_trans_key() for _ in range(120)]

    fixed_metrics = {}
    sdi_vals=[]; ksi_vals=[]; mi_vals=[]; chi_vals=[]; ent_vals=[]; mi_boot=[]
    for pt in plaintexts:
        sm,ss = symbol_diffusion_index(columnar_transposition_encrypt, pt, fixed_key, ALPH)
        sdi_vals.append(sm); ksi_vals.append(compute_ksi(columnar_transposition_encrypt, pt, fixed_key, perturb_trans_key(fixed_key)))
        C = columnar_transposition_encrypt(pt, fixed_key)
        mi_vals.append(compute_mi_from_plain_cipher_lists([pt],[C])); chi,H = compute_chi_entropy([C], ALPH)
        chi_vals.append(chi); ent_vals.append(H)
        mb,_ = bootstrap_mi(list(zip("".join(ch for ch in pt.upper() if ch.isalnum()), C[:len(pt)])), n_boot=150)
        mi_boot.append(mb)
    cdci_out = compute_cdci_for_cipher(columnar_transposition_encrypt, plaintexts, fixed_key, keyspace, alpha=alpha)
    fixed_metrics.update({
        'SDI_mean': np.mean(sdi_vals), 'KSI_mean': np.mean(ksi_vals), 'MI_mean': np.mean(mi_vals),
        'Chi2_mean': np.mean(chi_vals), 'Entropy_mean': np.mean(ent_vals), 'MI_boot_mean': np.mean(mi_boot),
        'C': cdci_out['C'], 'D': cdci_out['D'], 'CDCI': cdci_out['CDCI']
    })
    fixed_df = pd.DataFrame([fixed_metrics])

    random_keys = [random_trans_key() for _ in range(random_key_count)]
    rand_records=[]
    for rk in random_keys:
        sdi_vals=[]; ksi_vals=[]; mi_vals=[]; chi_vals=[]; ent_vals=[]; mi_boot=[]
        for pt in plaintexts:
            sm,ss = symbol_diffusion_index(columnar_transposition_encrypt, pt, rk, ALPH)
            sdi_vals.append(sm); ksi_vals.append(compute_ksi(columnar_transposition_encrypt, pt, rk, perturb_trans_key(rk)))
            C = columnar_transposition_encrypt(pt, rk)
            mi_vals.append(compute_mi_from_plain_cipher_lists([pt],[C])); chi,H = compute_chi_entropy([C], ALPH)
            chi_vals.append(chi); ent_vals.append(H)
            mb,_ = bootstrap_mi(list(zip("".join(ch for ch in pt.upper() if ch.isalnum()), C[:len(pt)])), n_boot=120)
            mi_boot.append(mb)
        cdci_out = compute_cdci_for_cipher(columnar_transposition_encrypt, plaintexts, rk, keyspace, alpha=alpha)
        rand_records.append({
            'key': rk, 'SDI_mean': np.mean(sdi_vals), 'KSI_mean': np.mean(ksi_vals),
            'MI_mean': np.mean(mi_vals), 'Chi2_mean': np.mean(chi_vals), 'Entropy_mean': np.mean(ent_vals),
            'MI_boot_mean': np.mean(mi_boot), 'C': cdci_out['C'], 'D': cdci_out['D'], 'CDCI': cdci_out['CDCI']
        })
    random_df = pd.DataFrame(rand_records)
    summary = pd.DataFrame({
        'mode':['fixed','random_mean'],
        'SDI_mean':[fixed_metrics['SDI_mean'], random_df['SDI_mean'].mean()],
        'KSI_mean':[fixed_metrics['KSI_mean'], random_df['KSI_mean'].mean()],
        'MI_mean':[fixed_metrics['MI_mean'], random_df['MI_mean'].mean()],
        'Entropy_mean':[fixed_metrics['Entropy_mean'], random_df['Entropy_mean'].mean()],
        'CDCI':[fixed_metrics['CDCI'], random_df['CDCI'].mean()],
        'C_index':[fixed_metrics['C'], random_df['C'].mean()],
        'D_index':[fixed_metrics['D'], random_df['D'].mean()]
    })

    #display_dataframe_to_user("Transposition - Fixed", fixed_df)
    #display_dataframe_to_user("Transposition - Random", random_df)
    #display_dataframe_to_user("Transposition - Summary", summary)

    # plots
    x = np.arange(len(summary)); width=0.25
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(x-width, summary['CDCI'], width, label='CDCI')
    ax.bar(x, summary['C_index'], width, label='C-index')
    ax.bar(x+width, summary['D_index'], width, label='D-index')
    ax.set_xticks(x); ax.set_xticklabels(summary['mode']); ax.set_ylabel('Score'); ax.set_title('Transposition: CDCI / C / D'); ax.legend(); plt.show()

    fixed_df.to_csv('./mnt/data/transposition_fixed.csv', index=False)
    random_df.to_csv('./mnt/data/transposition_random.csv', index=False)
    summary.to_csv('./mnt/data/transposition_summary.csv', index=False)
    print("Transposition saved to ./mnt/data/transposition_*.csv")
    return fixed_df, random_df, summary

# ----------------------------
# 5) ADFGVX cipher (Polybius + Columnar)
# ----------------------------
ALPH36 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
ADFGVX_CHARS = "ADFGVX"

def build_polybius_square(polykey):
    seen = []
    for ch in polykey.upper():
        if ch in ALPH36 and ch not in seen:
            seen.append(ch)
    for ch in ALPH36:
        if ch not in seen:
            seen.append(ch)
    coords = {}
    for idx, ch in enumerate(seen):
        coords[ch] = (ADFGVX_CHARS[idx//6], ADFGVX_CHARS[idx%6])
    return coords

def adfgvx_polybius_encode(text, polykey):
    coords = build_polybius_square(polykey)
    filtered = "".join(ch for ch in text.upper() if ch in ALPH36)
    return "".join(coords[ch] for ch in filtered)

def columnar_transposition_encrypt_string(s, key):
    n = len(key)
    rows = (len(s) + n - 1) // n
    padded = s + "X"*(rows*n - len(s))
    grid = [padded[i*n:(i+1)*n] for i in range(rows)]
    order = sorted(range(n), key=lambda i: key[i])
    return "".join("".join(row[j] for row in grid) for j in order)

def adfgvx_encrypt(plaintext, key_tuple):
    # key_tuple = (polykey36, trans_key)
    polykey, trans_key = key_tuple
    encoded = adfgvx_polybius_encode(plaintext, polykey)
    cipher = columnar_transposition_encrypt_string(encoded, trans_key)
    return cipher

def random_polykey36():
    chars = list(ALPH36); random.shuffle(chars); return "".join(chars)

def random_adfgvx_trans(min_len=4, max_len=8):
    length = random.randint(min_len, max_len)
    return "".join(random.choice(ALPH) for _ in range(length))

def perturb_adfgvx_key(key_tuple):
    pk, tk = key_tuple
    # small perturbation
    pk2 = list(pk); i,j = random.sample(range(36),2); pk2[i],pk2[j] = pk2[j],pk2[i]
    tk2 = list(tk)
    if len(tk2) >= 2:
        a,b = random.sample(range(len(tk2)),2); tk2[a],tk2[b] = tk2[b],tk2[a]
    return ("".join(pk2), "".join(tk2))

def evaluate_adfgvx(num_texts=5, length=200,
                    fixed_poly="PH0QG64MEA1YL2NOFDZXKR3CVS5W7BJ9UTI8",
                    fixed_trans="GERMAN", random_key_count=10, alpha=0.5):
    # prepare plaintexts (allow letters+digits)
    plaintexts = []
    for _ in range(num_texts):
        plaintexts.append("".join(random.choice(ALPH36) for _ in range(length)))
    plaintexts[0] = ("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG0123456789" * 4)[:length]

    keyspace = [(random_polykey36(), random_adfgvx_trans()) for _ in range(120)]

    fixed_metrics = {}
    sdi_vals=[]; ksi_vals=[]; mi_vals=[]; chi_vals=[]; ent_vals=[]; mi_boot=[]
    for pt in plaintexts:
        sm,ss = symbol_diffusion_index(lambda p,k: adfgvx_encrypt(p,k), pt, (fixed_poly, fixed_trans), ALPH36)
        sdi_vals.append(sm)
        pert = perturb_adfgvx_key((fixed_poly, fixed_trans)); ksi_vals.append(compute_ksi(lambda p,k: adfgvx_encrypt(p,k), pt, (fixed_poly, fixed_trans), pert))
        C = adfgvx_encrypt(pt, (fixed_poly, fixed_trans))
        # MI: align filtered plaintext and ciphertext (truncate)
        filtered_pt = "".join(ch for ch in pt.upper() if ch in ALPH36)
        pairs = list(zip(filtered_pt, C[:len(filtered_pt)]))
        mi_vals.append(mutual_information_pairs(pairs))
        chi,H = compute_chi_entropy([C], ADFGVX_CHARS); chi_vals.append(chi); ent_vals.append(H)
        mb,_ = bootstrap_mi(pairs, n_boot=150); mi_boot.append(mb)

    cdci_out = compute_cdci_for_cipher(lambda p,k: adfgvx_encrypt(p,k), plaintexts, (fixed_poly,fixed_trans), keyspace, alpha=alpha)
    fixed_metrics.update({
        'SDI_mean': np.mean(sdi_vals), 'KSI_mean': np.mean(ksi_vals), 'MI_mean': np.mean(mi_vals),
        'Chi2_mean': np.mean(chi_vals), 'Entropy_mean': np.mean(ent_vals), 'MI_boot_mean': np.mean(mi_boot),
        'C': cdci_out['C'], 'D': cdci_out['D'], 'CDCI': cdci_out['CDCI']
    })
    fixed_df = pd.DataFrame([fixed_metrics])

    random_keys = [(random_polykey36(), random_adfgvx_trans()) for _ in range(random_key_count)]
    rand_records=[]
    for rk in random_keys:
        sdi_vals=[]; ksi_vals=[]; mi_vals=[]; chi_vals=[]; ent_vals=[]; mi_boot=[]
        for pt in plaintexts:
            sm,ss = symbol_diffusion_index(lambda p,k: adfgvx_encrypt(p,k), pt, rk, ALPH36)
            sdi_vals.append(sm); ksi_vals.append(compute_ksi(lambda p,k: adfgvx_encrypt(p,k), pt, rk, perturb_adfgvx_key(rk)))
            C = adfgvx_encrypt(pt, rk)
            filtered_pt = "".join(ch for ch in pt.upper() if ch in ALPH36)
            pairs = list(zip(filtered_pt, C[:len(filtered_pt)]))
            mi_vals.append(mutual_information_pairs(pairs))
            chi,H = compute_chi_entropy([C], ADFGVX_CHARS); chi_vals.append(chi); ent_vals.append(H)
            mb,_ = bootstrap_mi(pairs, n_boot=120); mi_boot.append(mb)
        cdci_out = compute_cdci_for_cipher(lambda p,k: adfgvx_encrypt(p,k), plaintexts, rk, keyspace, alpha=alpha)
        rand_records.append({
            'poly': rk[0][:8], 'trans': rk[1], 'SDI_mean': np.mean(sdi_vals), 'KSI_mean': np.mean(ksi_vals),
            'MI_mean': np.mean(mi_vals), 'Chi2_mean': np.mean(chi_vals), 'Entropy_mean': np.mean(ent_vals),
            'MI_boot_mean': np.mean(mi_boot), 'C': cdci_out['C'], 'D': cdci_out['D'], 'CDCI': cdci_out['CDCI']
        })
    random_df = pd.DataFrame(rand_records)
    summary = pd.DataFrame({
        'mode':['fixed','random_mean'],
        'SDI_mean':[fixed_metrics['SDI_mean'], random_df['SDI_mean'].mean()],
        'KSI_mean':[fixed_metrics['KSI_mean'], random_df['KSI_mean'].mean()],
        'MI_mean':[fixed_metrics['MI_mean'], random_df['MI_mean'].mean()],
        'Entropy_mean':[fixed_metrics['Entropy_mean'], random_df['Entropy_mean'].mean()],
        'CDCI':[fixed_metrics['CDCI'], random_df['CDCI'].mean()],
        'C_index':[fixed_metrics['C'], random_df['C'].mean()],
        'D_index':[fixed_metrics['D'], random_df['D'].mean()]
    })

    #display_dataframe_to_user("ADFGVX - Fixed", fixed_df)
    #display_dataframe_to_user("ADFGVX - Random", random_df)
    #display_dataframe_to_user("ADFGVX - Summary", summary)

    # plots
    x = np.arange(len(summary)); width=0.25
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(x-width, summary['CDCI'], width, label='CDCI')
    ax.bar(x, summary['C_index'], width, label='C-index')
    ax.bar(x+width, summary['D_index'], width, label='D-index')
    ax.set_xticks(x); ax.set_xticklabels(summary['mode']); ax.set_ylabel('Score'); ax.set_title('ADFGVX: CDCI / C / D'); ax.legend(); plt.show()

    fixed_df.to_csv('./mnt/data/adfgvx_fixed.csv', index=False)
    random_df.to_csv('./mnt/data/adfgvx_random.csv', index=False)
    summary.to_csv('./mnt/data/adfgvx_summary.csv', index=False)
    print("ADFGVX saved to ./mnt/data/adfgvx_*.csv")
    return fixed_df, random_df, summary

# ----------------------------
# 6) Hill cipher (n x n)
# ----------------------------
class HillCipher:
    def __init__(self, key, n):
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.char_to_num = {char: i for i, char in enumerate(self.alphabet)}
        self.num_to_char = {i: char for i, char in enumerate(self.alphabet)}
        self.n = n
        self.key_matrix = self.getKeyMatrix(key, n)
        self.inverse_key_matrix = self.modMatInv(self.key_matrix, 26)

    def getKeyMatrix(self, key, n):
        k = 0
        keyMatrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                keyMatrix[i][j] = ord(key[k]) % 65
                k += 1
        return np.array(keyMatrix)

    def padding(self, plaintext):
        plaintext = plaintext.replace(" ", "").upper()
        pad_len = (self.n - len(plaintext) % self.n) % self.n
        plaintext += 'X' * pad_len
        return plaintext, pad_len

    def encrypt(self, plaintext):
        plaintext, pad_len = self.padding(plaintext)
        blocks = np.array([[self.char_to_num[char]] for char in plaintext])
        plaintext_blocks = [blocks[i:i+self.n] for i in range(0, len(blocks), self.n)]
        ciphertext = ""
        for block in plaintext_blocks:
            encrypted_block = np.dot(self.key_matrix, block) % 26
            for num in encrypted_block:
                ciphertext += self.num_to_char[num[0]]
        return ciphertext, pad_len

    def modMatInv(self, matrix, modulus):
        M = Matrix(matrix)
        det = int(M.det()) % modulus
        if math.gcd(det, modulus) != 1:
            raise ValueError("Key matrix not invertible modulo {}".format(modulus))
        det_inv = pow(det, -1, modulus)
        adjugate = M.adjugate()
        return np.array((det_inv * adjugate) % modulus).astype(int)

    def decrypt(self, ciphertext, pad_len=0):
        blocks = np.array([[self.char_to_num[char]] for char in ciphertext])
        cipher_blocks = [blocks[i:i+self.n] for i in range(0, len(blocks), self.n)]
        decrypted_text = ""
        for block in cipher_blocks:
            decrypted_block = np.dot(self.inverse_key_matrix, block) % 26
            for num in decrypted_block:
                decrypted_text += self.num_to_char[num[0]]
        if pad_len > 0:
            decrypted_text = decrypted_text[:-pad_len]
        return decrypted_text

def random_hill_key(n=3):
    attempts = 0
    while True:
        attempts += 1
        chars = [random.choice(ALPH) for _ in range(n*n)]
        key_str = "".join(chars)
        try:
            _ = HillCipher(key_str, n)
            return key_str
        except Exception:
            if attempts > 5000:
                raise RuntimeError("Failed to generate invertible Hill key after many attempts")

def perturb_hill_key(key_str, n=3, max_trials=200):
    base = list(key_str)
    for _ in range(max_trials):
        k = base[:]
        pos = random.randrange(len(k))
        k[pos] = random.choice([c for c in ALPH if c != k[pos]])
        cand = "".join(k)
        try:
            _ = HillCipher(cand, n)
            return cand
        except Exception:
            continue
    for _ in range(max_trials):
        k = base[:]
        i,j = random.sample(range(len(k)),2)
        k[i],k[j] = k[j],k[i]
        cand = "".join(k)
        try:
            _ = HillCipher(cand, n)
            return cand
        except Exception:
            continue
    raise RuntimeError("Could not find invertible perturbed key")

def hill_encrypt_wrapper(plaintext, key_str, n=3):
    hc = HillCipher(key_str, n)
    C, pad = hc.getCiphertext(plaintext)
    return C

def evaluate_hill(n=3, num_texts=5, length=200, fixed_key="GYBNQKURP", random_key_count=10, alpha=0.5):
    plaintexts = [ "".join(random.choice(ALPH) for _ in range(length)) for _ in range(num_texts) ]
    plaintexts[0] = ("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG" * 6)[:length]
    keyspace = [random_hill_key(n) for _ in range(80)]

    fixed_metrics = {}
    sdi_vals=[]; ksi_vals=[]; mi_vals=[]; chi_vals=[]; ent_vals=[]; mi_boot=[]
    for pt in plaintexts:
        sm,ss = symbol_diffusion_index(lambda p,k: hill_encrypt_wrapper(p,k,n), pt, fixed_key, ALPH)
        sdi_vals.append(sm)
        try:
            pert = perturb_hill_key(fixed_key, n=n)
        except Exception:
            pert = fixed_key
        ksi_vals.append(compute_ksi(lambda p,k: hill_encrypt_wrapper(p,k,n), pt, fixed_key, pert))
        C = hill_encrypt_wrapper(pt, fixed_key, n)
        mi_vals.append(compute_mi_from_plain_cipher_lists([pt],[C])); chi,H = compute_chi_entropy([C], ALPH)
        chi_vals.append(chi); ent_vals.append(H)
        pairs = list(zip("".join(ch for ch in pt.upper() if ch.isalnum()), C[:len(pt)]))
        mb,_ = bootstrap_mi(pairs, n_boot=150); mi_boot.append(mb)

    cdci_out = compute_cdci_for_cipher(lambda p,k: hill_encrypt_wrapper(p,k,n), plaintexts, fixed_key, keyspace, alpha=alpha)
    fixed_metrics.update({
        'SDI_mean': np.mean(sdi_vals), 'KSI_mean': np.mean(ksi_vals), 'MI_mean': np.mean(mi_vals),
        'Chi2_mean': np.mean(chi_vals), 'Entropy_mean': np.mean(ent_vals), 'MI_boot_mean': np.mean(mi_boot),
        'C': cdci_out['C'], 'D': cdci_out['D'], 'CDCI': cdci_out['CDCI']
    })
    fixed_df = pd.DataFrame([fixed_metrics])

    random_keys = [random_hill_key(n) for _ in range(random_key_count)]
    rand_records=[]
    for rk in random_keys:
        sdi_vals=[]; ksi_vals=[]; mi_vals=[]; chi_vals=[]; ent_vals=[]; mi_boot=[]
        for pt in plaintexts:
            sm,ss = symbol_diffusion_index(lambda p,k: hill_encrypt_wrapper(p,k,n), pt, rk, ALPH)
            sdi_vals.append(sm)
            try:
                pert = perturb_hill_key(rk, n=n)
            except Exception:
                pert = rk
            ksi_vals.append(compute_ksi(lambda p,k: hill_encrypt_wrapper(p,k,n), pt, rk, pert))
            C = hill_encrypt_wrapper(pt, rk, n)
            mi_vals.append(compute_mi_from_plain_cipher_lists([pt],[C])); chi,H = compute_chi_entropy([C], ALPH)
            chi_vals.append(chi); ent_vals.append(H)
            pairs = list(zip("".join(ch for ch in pt.upper() if ch.isalnum()), C[:len(pt)]))
            mb,_ = bootstrap_mi(pairs, n_boot=120); mi_boot.append(mb)
        cdci_out = compute_cdci_for_cipher(lambda p,k: hill_encrypt_wrapper(p,k,n), plaintexts, rk, keyspace, alpha=alpha)
        rand_records.append({
            'key': rk, 'SDI_mean': np.mean(sdi_vals), 'KSI_mean': np.mean(ksi_vals),
            'MI_mean': np.mean(mi_vals), 'Chi2_mean': np.mean(chi_vals), 'Entropy_mean': np.mean(ent_vals),
            'MI_boot_mean': np.mean(mi_boot), 'C': cdci_out['C'], 'D': cdci_out['D'], 'CDCI': cdci_out['CDCI']
        })
    random_df = pd.DataFrame(rand_records)
    summary = pd.DataFrame({
        'mode':['fixed','random_mean'],
        'SDI_mean':[fixed_metrics['SDI_mean'], random_df['SDI_mean'].mean()],
        'KSI_mean':[fixed_metrics['KSI_mean'], random_df['KSI_mean'].mean()],
        'MI_mean':[fixed_metrics['MI_mean'], random_df['MI_mean'].mean()],
        'Entropy_mean':[fixed_metrics['Entropy_mean'], random_df['Entropy_mean'].mean()],
        'CDCI':[fixed_metrics['CDCI'], random_df['CDCI'].mean()],
        'C_index':[fixed_metrics['C'], random_df['C'].mean()],
        'D_index':[fixed_metrics['D'], random_df['D'].mean()]
    })

    #display_dataframe_to_user("Hill - Fixed", fixed_df)
    #display_dataframe_to_user("Hill - Random", random_df)
    #display_dataframe_to_user("Hill - Summary", summary)

    # plots
    x = np.arange(len(summary)); width=0.25
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(x-width, summary['CDCI'], width, label='CDCI')
    ax.bar(x, summary['C_index'], width, label='C-index')
    ax.bar(x+width, summary['D_index'], width, label='D-index')
    ax.set_xticks(x); ax.set_xticklabels(summary['mode']); ax.set_ylabel('Score'); ax.set_title('Hill: CDCI / C / D'); ax.legend(); plt.show()

    fixed_df.to_csv('./mnt/data/hill_fixed.csv', index=False)
    random_df.to_csv('./mnt/data/hill_random.csv', index=False)
    summary.to_csv('./mnt/data/hill_summary.csv', index=False)
    print("Hill saved to ./mnt/data/hill_*.csv")
    return fixed_df, random_df, summary

# ----------------------------
# End of file
# ----------------------------
# Usage examples (call any you'd like):
evaluate_shift()
evaluate_substitution()
evaluate_vigenere()
evaluate_transposition()
evaluate_adfgvx()
evaluate_hill()
