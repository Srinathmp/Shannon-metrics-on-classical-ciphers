import numpy as np
import random
from collections import Counter
import math

def measure_positional_diffusion(cipher_object, plaintext, num_trials=50):
    """
    Measures the positional predictability of changes in ciphertext.
    A lower score indicates more predictable changes (poor diffusion).
    """
    original_ciphertext, _ = cipher_object.encrypt(plaintext)
    cipher_len = len(original_ciphertext)
    
    changed_positions_all = []
    
    for _ in range(num_trials):
        # Create an incremental plaintext change
        modified_plaintext = list(plaintext)
        pos = random.randint(0, len(plaintext) - 1)
        modified_plaintext[pos] = random.choice(cipher_object.alphabet)
        modified_plaintext = "".join(modified_plaintext)
        
        try:
            modified_ciphertext, _ = cipher_object.encrypt(modified_plaintext)
        except Exception:
            continue
        
        changed_positions = [i for i, (c1, c2) in enumerate(zip(original_ciphertext, modified_ciphertext)) if c1 != c2]
        changed_positions_all.extend(changed_positions)
    
    if not changed_positions_all:
        return 0.0 # No changes were detected
    
    # Calculate a measure of distribution (e.g., variance of position indices)
    # A high variance indicates changes are scattered, low variance indicates clustering
    variance = np.var(changed_positions_all)
    
    # Normalize the score (you can adjust this formula)
    max_possible_variance = (cipher_len**2 - 1) / 12 # Variance of a uniform distribution
    normalized_score = variance / max_possible_variance
    
    return normalized_score

# Example Usage
# hill_cipher_obj = HillCipher(key="YOURKEY", n=3)
# score = measure_positional_diffusion(hill_cipher_obj, "ATTACKATDAWN")
# print(f"Positional Diffusion Score: {score:.2f}")

def measure_value_confusion(cipher_object, plaintext, num_trials=50):
    """
    Measures the predictability of the character values that change.
    A lower score indicates more predictable changes (poor confusion).
    """
    original_ciphertext, _ = cipher_object.encrypt(plaintext)
    value_differences_all = []
    
    original_key = cipher_object.key
    
    for _ in range(num_trials):
        # Create an incremental key change
        modified_key = list(original_key)
        pos = random.randint(0, len(original_key) - 1)
        modified_key[pos] = random.choice(cipher_object.alphabet)
        modified_key = "".join(modified_key)
        
        try:
            modified_cipher_obj = HillCipher(key=modified_key, n=cipher_object.n)
            modified_ciphertext, _ = modified_cipher_obj.encrypt(plaintext)
        except (ValueError, IndexError):
            continue
            
        # Calculate the numerical difference for changed characters
        diffs = [
            abs(cipher_object.char_to_num[c1] - cipher_object.char_to_num[c2])
            for c1, c2 in zip(original_ciphertext, modified_ciphertext) if c1 != c2
        ]
        value_differences_all.extend(diffs)
    
    if not value_differences_all:
        return 0.0 # No changes were detected
        
    # Calculate a measure of uniformity or randomness, like normalized entropy
    # A high entropy indicates a wide, unpredictable range of changes
    counts = Counter(value_differences_all)
    probabilities = np.array(list(counts.values())) / len(value_differences_all)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    max_entropy = np.log2(len(set(value_differences_all)))
    if max_entropy == 0:
        return 0.0
        
    normalized_entropy = entropy / max_entropy
    
    return normalized_entropy

# Example Usage
# hill_cipher_obj = HillCipher(key="YOURKEY", n=3)
# score = measure_value_confusion(hill_cipher_obj, "QUANTIFYCONFUSION")
# print(f"Character Value Confusion Score: {score:.2f}")