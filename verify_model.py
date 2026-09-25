#!/usr/bin/env python3
"""Sanity-check the exported quantized transformer before shipping it into index.html."""
import base64, json
import numpy as np
import train_ai as T

M = json.load(open("model.json"))

def W(o):
    out = len(o["b"])
    q = np.frombuffer(base64.b64decode(o["w"]), dtype=np.int8).astype(np.float32)
    inn = q.size // out
    return q.reshape(out, inn) * o["s"], np.array(o["b"], np.float32)

D, H, FF = M["dModel"], M["heads"], M["ff"]
HD = D // H
S, SD = M["segs"], M["segDim"]
EPS = 1e-5

projW, projB = W(M["proj"])
pos = np.array(M["pos"], np.float32)          # (S+1, D)
cls = np.array(M["cls"], np.float32)
headW, headB = W(M["head"])

BLK = []
for b in M["blocks"]:
    qkvW, qkvB = W(b["qkv"])                 # (3D, D)
    outW, outB = W(b["out"])
    w1, b1 = W(b["w1"])
    w2, b2 = W(b["w2"])
    BLK.append(dict(
        ln1g=np.array(b["ln1"]["g"], np.float32), ln1b=np.array(b["ln1"]["b"], np.float32),
        qkvW=qkvW, qkvB=qkvB, outW=outW, outB=outB,
        ln2g=np.array(b["ln2"]["g"], np.float32), ln2b=np.array(b["ln2"]["b"], np.float32),
        w1=w1, b1=b1, w2=w2, b2=b2))

def ln(x, g, b):  # x: (n, D)
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + EPS) * g + b

def fwd(X):       # X: (B, 2608)
    B = X.shape[0]
    seg = X.reshape(B, S, SD)
    toks = seg @ projW.T + projB + pos[1:]
    clsv = cls.reshape(1, 1, D) + pos[:1]
    h = np.concatenate([clsv.repeat(B, 0), toks], axis=1)   # (B, S+1, D)
    n = S + 1
    for L in BLK:
        t = ln(h, L["ln1g"], L["ln1b"])
        qkv = t @ L["qkvW"].T + L["qkvB"]                   # (B, n, 3D)
        q, k, v = qkv[:, :, :D], qkv[:, :, D:2 * D], qkv[:, :, 2 * D:]
        q = q.reshape(B, n, H, HD).transpose(0, 2, 1, 3)   # (B,H,n,HD)
        k = k.reshape(B, n, H, HD).transpose(0, 2, 1, 3)
        v = v.reshape(B, n, H, HD).transpose(0, 2, 1, 3)
        att = q @ k.transpose(0, 1, 3, 2) / np.sqrt(HD).astype(np.float32)
        att = att - att.max(-1, keepdims=True)
        e = np.exp(att)
        p = (e / e.sum(-1, keepdims=True)).astype(np.float32)
        o = (p @ v).transpose(0, 2, 1, 3).reshape(B, n, D)
        h = h + o @ L["outW"].T + L["outB"]
        t = ln(h, L["ln2g"], L["ln2b"])
        ff = np.maximum(t @ L["w1"].T + L["b1"], 0)
        h = h + ff @ L["w2"].T + L["b2"]
    c = ln(h[:, 0], np.array(M["lnF"]["g"], np.float32), np.array(M["lnF"]["b"], np.float32))
    return 1 / (1 + np.exp(-(c @ headW.T + headB)))

assert M["params"] == 8_269_972, "unexpected parameter count: %s" % M["params"]
assert [len(M["blocks"])] == [6] and M["layers"] == 6 and M["heads"] == 8

Xv, Yv = T.gen_batch(4000)
p = np.concatenate([fwd(Xv[i:i+500]) for i in range(0, 4000, 500)], axis=0)
ok = True
for c, label in enumerate(T.LABELS):
    acc = float(((p[:, c] > 0.5) == (Yv[:, c] > 0.5)).mean())
    print("quantized accuracy %-22s = %.4f" % (label, acc))
    ok &= acc >= 0.98
if not ok:
    raise SystemExit("MODEL CHECK FAILED")
print("MODEL CHECK PASSED")
