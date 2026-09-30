"""
Mimo AI // Standalone Forward Pass & Attention Visualization Demo
Pure NumPy implementation demonstrating causal self-attention mechanics.
Author: MD. Raisul Islam Mobin | mobin.tech
"""
import numpy as np

# 32 ASCII Character Vocabulary
CHARS = " abcdefghijklmnopqrstuvwxyz.,?!\n"
STOI = {ch: i for i, ch in enumerate(CHARS)}
ITOS = {i: ch for i, ch in enumerate(CHARS)}

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def main():
    print("=" * 65)
    print("[*] Mimo AI // Self-Attention & Forward Pass Demo")
    print("    Mathematical First-Principles in Pure NumPy")
    print("=" * 65)

    prompt = "who is mobin"
    tokens = [STOI.get(c, 0) for c in prompt.lower()][:16]
    T = len(tokens)
    d = 16

    print(f"\n[1] Input Sequence: {prompt!r}")
    print(f"    Token IDs ({T} tokens): {tokens}")

    # Simulated initialized weights
    np.random.seed(42)
    tok_emb = np.random.randn(32, d) * 0.05
    pos_emb = np.random.randn(16, d) * 0.05
    Wq = np.random.randn(d, d) * 0.05
    Wk = np.random.randn(d, d) * 0.05
    Wv = np.random.randn(d, d) * 0.05

    # 1. Embeddings
    x = tok_emb[tokens] + pos_emb[:T]
    print(f"\n[2] Embedding Space: Tensor Shape ({T}, {d})")

    # 2. Causal Self-Attention
    q = x @ Wq
    k = x @ Wk
    v = x @ Wv

    scores = (q @ k.T) / np.sqrt(d)
    causal_mask = np.triu(np.ones((T, T)) * -1e9, k=1)
    masked_scores = scores + causal_mask
    attn_weights = softmax(masked_scores)

    print(f"\n[3] Causal Attention Matrix (Strict Upper-Triangular Masking):")
    print("    Rows attend only to previous and current tokens:")
    for i in range(min(5, T)):
        row_str = " ".join(f"{w:.2f}" for w in attn_weights[i, :min(5, T)])
        print(f"    Token '{prompt[i]}' -> [{row_str} ...]")

    # 3. Contextual Output
    out = attn_weights @ v
    print(f"\n[4] Contextualized Attention Output Shape: {out.shape}")
    print("\n[OK] First-principles forward pass completed with 0 dependencies.")

if __name__ == "__main__":
    main()
