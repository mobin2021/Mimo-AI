"""
Mimo AI // Exact Parameter Audit
Verifies the analytical 1,824-parameter budget.
Author: MD. Raisul Islam Mobin | mobin.tech
"""
import numpy as np

def audit():
    print("=" * 60)
    print("[*] Mimo AI // Parameter Allocation Audit")
    print("=" * 60)
    
    vocab_size = 32
    d = 16
    T = 16
    
    breakdown = {
        "Token Embeddings (32 x 16)": vocab_size * d,
        "Positional Embeddings (16 x 16)": T * d,
        "Attention Query Weights Wq (16 x 16)": d * d,
        "Attention Key Weights Wk (16 x 16)": d * d,
        "Attention Value Weights Wv (16 x 16)": d * d,
        "FFN Bottleneck W1 (16 x 8)": d * 8,
        "FFN Bias b1 (1 x 8)": 8,
        "FFN Projection W2 (8 x 16)": 8 * d,
        "FFN Bias b2 (1 x 16)": d,
        "Attention Bias a_bias (1 x 8)": 8,
    }
    
    total = sum(breakdown.values())
    
    for name, count in breakdown.items():
        print(f"  {name:<40} : {count:>5d} params")
        
    print("-" * 60)
    print(f"  {'TOTAL MODEL BUDGET':<40} : {total:>5d} params")
    print("=" * 60)
    
    assert total == 1824, f"Budget error: expected 1824, got {total}"
    print("[PASS] Strict parameter budget verified: Exactly 1,824 parameters.\n")

if __name__ == "__main__":
    audit()
