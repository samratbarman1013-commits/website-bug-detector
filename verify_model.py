#!/usr/bin/env python3
"""Sanity-check the exported quantized model before shipping it into index.html."""
import base64, json
import numpy as np
import train_ai as T

M = json.load(open("model.json"))

def deq(L):
    q = np.frombuffer(base64.b64decode(L["w"]), dtype=np.int8).astype(np.float32)
    return q.reshape(L["out"], L["in"]) * L["s"], np.array(L["b"], np.float32)

Ws = [deq(L) for L in M["layers"]]

def fwd(X):
    h = X
    for i, (W, b) in enumerate(Ws):
        z = h @ W.T + b
        h = 1 / (1 + np.exp(-z)) if i == len(Ws) - 1 else np.maximum(z, 0)
    return h

assert M["params"] == 1_500_548, "unexpected parameter count: %s" % M["params"]
assert [L["in"] for L in M["layers"]] == [2608, 512, 256, 128]

Xv, Yv = T.gen_batch(4000)
p = fwd(Xv)
ok = True
for c, label in enumerate(T.LABELS):
    acc = float(((p[:, c] > 0.5) == (Yv[:, c] > 0.5)).mean())
    print("quantized accuracy %-22s = %.4f" % (label, acc))
    ok &= acc >= 0.98
if not ok:
    raise SystemExit("MODEL CHECK FAILED")
print("MODEL CHECK PASSED")
