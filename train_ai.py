#!/usr/bin/env python3
"""
BugHunter AI risk model — v4.5.  REAL TRANSFORMER.
Architecture : pre-LN transformer encoder
               input 2608 -> 8 segments x 326 (+1 CLS) -> token proj 336
               6 encoder layers: 8-head self-attention (d=336) + FFN 336->1344->336
               final LayerNorm on CLS -> 4 sigmoid heads
Parameters   : 8,269,972
Inputs       : 2560 hashed bag-of-tokens (FNV-1a % 2560) + 48 numeric features
Outputs      : multi-label threat probabilities
               [phishing/scam, malware/compromise, outdated stack, poor quality]
Training     : PyTorch (CPU) + AdamW, BCE loss, on a heuristic-derived synthetic
               corpus (expert rules distilled into a neural net) with noise tokens
               and missing-data augmentation.
Usage         : python3 train_ai.py <steps_this_round> [resume]
               (validates + exports model.json when cumulative steps >= 250)
"""
import json, base64, time, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

rng = np.random.default_rng(42)
torch.manual_seed(42)
torch.set_num_threads(2)

HASH_DIM, N_FEAT = 2560, 48
IN_DIM = HASH_DIM + N_FEAT          # 2608
SEGS, SEG_DIM = 8, 326              # 8 x 326 = 2608
D_MODEL, N_HEADS, N_LAYERS, FF = 336, 8, 6, 1344
TOTAL_STEPS = 250
LABELS = ["phishing_scam", "malware_compromised", "outdated_insecure", "poor_quality"]

def fnv1a(s):
    h = 0x811c9dc5
    for ch in s.encode():
        h ^= ch
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h

BENIGN = ["home","about","contact","services","products","blog","news","privacy","terms","menu",
"search","email","phone","company","team","copyright","careers","newsletter","subscribe",
"gallery","portfolio","client","project","story","mission","values","reviews","testimonials",
"faq","shipping","returns","cart","checkout","store","shop","download","app","features",
"pricing","plans","docs","guide","tutorial","resource","events","webinar","press","design",
"creative","digital","solution","platform","tools","start","learn","read","latest","post",
"comment","share","category","archive","author","dashboard","profile","settings","help",
"support","follow","social","media","work","agency","studio","software","cloud","data"]

PHISH = ["verify","verified","account","secure","security","update","billing","wallet","confirm",
"suspended","unlimited","winner","prize","gift","card","unlock","limited","offer","password",
"banking","credential","validate","identity","otp","kyc","invoice","payment","refund","urgent",
"warning","notice","locked","access","restore","recover","immediately","click","signin",
"recovery","code","bonus","cash","reward","claim","expired","deactivated","unusual","activity",
"protect","safeguard"]

BRAND = ["paypal","amazon","apple","netflix","instagram","facebook","google","microsoft","binance",
"whatsapp","gmail","icloud","steampowered","roblox","telegram","chase","wells","fargo","hsbc",
"sbi","hdfc","paytm","phonepe","flipkart"]

MAL = ["eval","atob","unescape","fromcharcode","charcodeat","document","write","iframe",
"cryptonight","coinhive","stratum","miner","pool","worker","wasm","popunder","casino","adult",
"download","crack","keygen","torrent","serial","warez","pharma","viagra","porn","betting",
"slots","poker","replica","backlink","hack","cheat","mod","apk","patch","activator","loader",
"dropper","shell","c99","webshell","base64","decodeuricomponent","settimeout","anonymous"]

OUTD = ["jquery","mootools","prototype","scriptaculous","bootstrap","wpcontent","wpincludes",
"plugins","themes","legacy","flash","shockwave","silverlight","swf","marquee","frameset",
"spacer","gif","compat","xhtml","dojo","flashplayer","activex"]

QUAL = ["untitled","default","enter","welcome","page","test","temp","draft","lorem","ipsum",
"sample","example","page2","copy","index","template","demo"]

VOCABS = [("benign", BENIGN, 0.55), ("phish", PHISH, 0.30), ("brand", BRAND, 0.40),
          ("mal", MAL, 0.30), ("outd", OUTD, 0.35), ("qual", QUAL, 0.35)]
ALL_TOKENS = sorted(set().union(*[set(v[1]) for v in VOCABS]))
VOCAB_IDX = {name: np.array([fnv1a(t) % HASH_DIM for t in toks]) for name, toks, _ in VOCABS}

def gen_batch(n):
    """Vectorized synthetic sample generator. Mirrors the JS feature contract."""
    R = lambda: rng.random(n)
    phish = R() < 0.12
    malw  = R() < np.where(phish, 0.35, 0.10)
    outd  = R() < 0.22
    qual  = R() < np.where(outd, 0.50, 0.25)

    # ---------------- URL base ----------------
    url_len = rng.uniform(20, 90, n)
    host_dig = rng.uniform(0, 0.08, n)
    host_hyph = rng.integers(0, 2, n).astype(float)
    r = R(); subd = np.where(r < .5, 0, np.where(r < .83, 1, 2)).astype(float)
    ip = np.zeros(n); sus_tld = np.zeros(n); puny = np.zeros(n)
    https = (R() < 0.85).astype(float)
    path_depth = rng.integers(0, 4, n).astype(float)
    query_len = rng.uniform(0, 40, n)
    at = np.zeros(n); cred_url = np.zeros(n)

    # ---------------- page base ----------------
    has_title = (R() < .93).astype(float); title_len = rng.uniform(8, 70, n)
    has_desc = (R() < .85).astype(float); desc_len = rng.uniform(60, 230, n)
    viewport = (R() < .87).astype(float); charset = (R() < .95).astype(float)
    lang = (R() < .85).astype(float); favicon = (R() < .8).astype(float)
    r = R(); h1 = np.where(r < .2, 0, np.where(r < .8, 1, 2)).astype(float)
    imgs = rng.integers(3, 50, n).astype(float); noalt = rng.uniform(0, .15, n)
    inline_styles = rng.integers(0, 15, n).astype(float)
    scripts = rng.integers(2, 25, n).astype(float)
    ext_scripts = np.maximum(0, scripts - rng.integers(0, 3, n)).astype(float)
    http_ext = rng.uniform(0, .05, n)
    r = R(); iframes = np.where(r < .8, 0, rng.integers(1, 3, n)).astype(float)
    forms = rng.integers(0, 3, n).astype(float)
    pw_form = ((forms > 0) & (R() < 0.2)).astype(float)
    form_http = np.zeros(n)
    r = R(); mixed = np.where(r < .95, 0, rng.integers(1, 3, n)).astype(float)
    handlers = rng.integers(0, 8, n).astype(float)
    has_jq = (R() < .4).astype(float); jq_ver = np.where(has_jq > 0, 3.0, 0.0)
    deprecated = np.zeros(n)
    blank_ratio = rng.uniform(.1, .4, n)
    r = R(); ev = np.where(r < .95, 0, 1).astype(float)
    obfs = rng.integers(0, 2, n).astype(float)
    long_hex = np.zeros(n); kw = np.zeros(n); miner = np.zeros(n)
    csp = (R() < .25).astype(float); hsts = (R() < .45).astype(float); xfo = (R() < .4).astype(float)
    meta_refresh = (R() < .01).astype(float)
    tok_counts = {"phish": np.zeros(n, bool), "brand": np.zeros(n, bool),
                  "mal": np.zeros(n, bool), "outd": np.zeros(n, bool),
                  "qual": np.zeros(n, bool)}

    # ---------------- class overrides ----------------
    if phish.any():
        m = phish
        url_len[m] = rng.uniform(60, 200, m.sum())
        host_hyph[m] = rng.integers(1, 5, m.sum())
        r = R(); subd[m] = np.where(r[m] < .5, 1, np.where(r[m] < .83, 2, 3))
        ip[m] = R()[m] < .15; sus_tld[m] = R()[m] < .25; puny[m] = R()[m] < .05
        https[m] = R()[m] < .6; cred_url[m] = R()[m] < .7
        has_title[m] = R()[m] < .8; title_len[m] = rng.uniform(10, 55, m.sum())
        has_desc[m] = R()[m] < .5; desc_len[m] = rng.uniform(0, 120, m.sum())
        viewport[m] = R()[m] < .5; charset[m] = R()[m] < .7; lang[m] = R()[m] < .5; favicon[m] = R()[m] < .4
        r = R(); h1[m] = np.where(r[m] < .5, 0, 1)
        imgs[m] = rng.integers(0, 8, m.sum()); noalt[m] = rng.uniform(.3, .9, m.sum())
        forms[m] = rng.integers(1, 3, m.sum())
        pw_form[m] = R()[m] < .75
        form_http[m] = (pw_form[m] > 0.5) & (R()[m] < .25)
        meta_refresh[m] = R()[m] < .35
        kw[m] = rng.integers(1, 6, m.sum())
        tok_counts["phish"] |= phish; tok_counts["brand"] |= phish
    if malw.any():
        m = malw
        ev[m] = rng.integers(2, 10, m.sum()); obfs[m] = rng.integers(2, 10, m.sum())
        long_hex[m] = R()[m] < .6; miner[m] = R()[m] < .3
        iframes[m] = rng.integers(2, 6, m.sum()); handlers[m] = rng.integers(10, 60, m.sum())
        scripts[m] = rng.integers(15, 45, m.sum())
        ext_scripts[m] = np.maximum(0, scripts[m] - rng.integers(0, 3, m.sum()))
        http_ext[m] = rng.uniform(.2, .9, m.sum()); mixed[m] = rng.integers(1, 6, m.sum())
        tok_counts["mal"] |= malw
    if outd.any():
        m = outd
        has_jq[m] = R()[m] < .8
        jq_ver[m] = np.where(has_jq[m] > 0, rng.uniform(1.0, 1.9, m.sum()), 0.0)
        deprecated[m] = rng.integers(1, 5, m.sum())
        mixed[m] = np.maximum(mixed[m], rng.integers(2, 8, m.sum()))
        http_ext[m] = np.maximum(http_ext[m], rng.uniform(.3, .8, m.sum()))
        viewport[m] = R()[m] < .5; charset[m] = R()[m] < .7; favicon[m] = R()[m] < .5
        inline_styles[m] = rng.integers(10, 50, m.sum())
        hsts[m] = R()[m] < .05; csp[m] = R()[m] < .08
        tok_counts["outd"] |= outd
    if qual.any():
        m = qual
        has_title[m] = R()[m] < .45; title_len[m] = rng.uniform(0, 30, m.sum())
        has_desc[m] = R()[m] < .3; desc_len[m] = rng.uniform(0, 80, m.sum())
        viewport[m] = R()[m] < .35; charset[m] = R()[m] < .6; lang[m] = R()[m] < .25; favicon[m] = R()[m] < .3
        r = R(); h1[m] = np.where(r[m] < .5, 0, np.where(r[m] < .75, 3, 4))
        imgs[m] = rng.integers(0, 20, m.sum()); noalt[m] = rng.uniform(.3, .8, m.sum())
        inline_styles[m] = rng.integers(20, 60, m.sum())
        tok_counts["qual"] |= qual

    # ---------------- feature block (48, order = JS contract) ----------------
    f = np.stack([
        np.minimum(url_len / 200, 1), rng.uniform(6, 20, n) / 60,
        host_dig, np.minimum(host_hyph / 6, 1), np.minimum(subd / 5, 1),
        ip, sus_tld, puny, https, np.minimum(path_depth / 8, 1),
        np.minimum(query_len / 100, 1), at, cred_url,
        has_title, np.minimum(title_len / 80, 1), has_desc, np.minimum(desc_len / 300, 1),
        viewport, charset, lang, favicon, np.minimum(h1 / 5, 1),
        np.minimum(imgs / 100, 1), noalt, np.minimum(inline_styles / 50, 1),
        np.minimum(scripts / 40, 1), np.minimum(iframes / 5, 1), np.minimum(forms / 5, 1),
        pw_form, form_http, np.minimum(mixed / 10, 1), np.minimum(handlers / 50, 1),
        has_jq, np.where(jq_ver > 0, jq_ver / 3, 0), np.minimum(deprecated / 5, 1),
        blank_ratio, np.minimum(ev / 10, 1), np.minimum(obfs / 10, 1), long_hex,
        np.minimum(kw / 8, 1), miner, np.minimum(ext_scripts / 20, 1), http_ext,
        csp, hsts, xfo, meta_refresh, np.minimum(pw_form, 1),
    ], axis=1).astype(np.float32)
    assert f.shape == (n, N_FEAT)

    # ---------------- hashed token block ----------------
    X = np.zeros((n, IN_DIM), np.float32)
    k = rng.integers(20, 55, n)
    draw = rng.integers(0, len(BENIGN), (n, 55))
    mask = rng.random((n, 55)) * 55 < k[:, None]
    BI = VOCAB_IDX["benign"]
    rows, cols = np.nonzero(mask)
    X[rows, BI[draw[rows, cols]]] = 1.0
    for name, toks, p in VOCABS:
        if name == "benign":
            continue
        active = tok_counts[name]
        if not active.any():
            continue
        sel = (rng.random((n, len(toks))) < p) & active[:, None]
        r_, c_ = np.nonzero(sel)
        X[r_, VOCAB_IDX[name][c_]] = 1.0
    cnt = rng.integers(0, 25, n)
    total = int(cnt.sum())
    if total:
        nidx = rng.integers(0, HASH_DIM, total)
        ridx = np.repeat(np.arange(n), cnt)
        X[ridx, nidx] = 1.0
    X[:, HASH_DIM:] = f
    miss = rng.random(n) < 0.03
    X[miss, HASH_DIM + 13: HASH_DIM + N_FEAT] = 0.5

    Y = np.stack([phish, malw, outd, qual], axis=1).astype(np.float32)
    return X, Y

# ---------------- transformer model ------------------------------------------------

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.fc1 = nn.Linear(D_MODEL, FF)
        self.fc2 = nn.Linear(FF, D_MODEL)

    def forward(self, x):
        t = self.ln1(x)
        a, _ = self.attn(t, t, t, need_weights=False)
        x = x + a
        x = x + self.fc2(F.relu(self.fc1(self.ln2(x))))
        return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(SEG_DIM, D_MODEL)
        self.cls = nn.Parameter(torch.zeros(D_MODEL))
        self.pos = nn.Parameter(torch.zeros(1, SEGS + 1, D_MODEL))
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.lnF = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, 4)

    def forward(self, x):                    # x: (B, 2608)
        B = x.shape[0]
        seg = x.view(B, SEGS, SEG_DIM)
        toks = self.proj(seg) + self.pos[:, 1:, :]
        cls = self.cls.view(1, 1, D_MODEL).expand(B, 1, D_MODEL) + self.pos[:, :1, :]
        h = torch.cat([cls, toks], dim=1)
        for b in self.blocks:
            h = b(h)
        return self.head(self.lnF(h[:, 0]))  # logits (B, 4)

def count_params(net):
    return sum(p.numel() for p in net.parameters())

def train_round(net, opt, steps, t_start, batch=64):
    t0 = time.time()
    for step in range(t_start + 1, t_start + steps + 1):
        X, Y = gen_batch(batch)
        xb = torch.from_numpy(X)
        yb = torch.from_numpy(Y)
        logits = net(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            with torch.no_grad():
                p = torch.sigmoid(logits)
                acc = ((p > 0.5) == yb).float().mean().item()
            print(f"step {step:4d}  loss {loss.item():.5f}  acc {acc:.3f}  ({time.time()-t0:.0f}s)", flush=True)

def validate(net, N=4000):
    net.eval()
    Xv, Yv = gen_batch(N)
    with torch.no_grad():
        p = torch.sigmoid(net(torch.from_numpy(Xv))).numpy()
    for c in range(4):
        order = np.argsort(p[:, c])
        ranks = np.empty(N); ranks[order] = np.arange(1, N + 1)
        pos = Yv[:, c] == 1
        n1, n0 = pos.sum(), (~pos).sum()
        auc = float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")
        print(f"val AUC {LABELS[c]:22s} = {auc:.4f}", flush=True)
    net.train()
    return Xv, Yv, p

def saliency(net):
    """Per-class mean input-gradient over class-positive synthetic samples."""
    out = []
    for c in range(4):
        X, Y = gen_batch(8192)
        sel = np.where(Y[:, c] > 0.5)[0][:256]
        if len(sel) < 32:
            sel = np.where(Y[:, c] >= 0)[0][:256]
        xb = torch.from_numpy(X[sel]).requires_grad_(True)
        s = torch.sigmoid(net(xb))[:, c].sum()
        net.zero_grad()
        s.backward()
        g = xb.grad.detach().numpy().mean(axis=0)
        out.append(g.astype(np.float32))
    return out

def q_int8(W):
    s = float(np.abs(W).max() / 127.0) or 1e-12
    q = np.clip(np.round(W / s), -127, 127).astype(np.int8)
    return q, s

def b64(arr):
    return base64.b64encode(arr.tobytes()).decode("ascii")

def qt(W):  # torch tensor -> quantized dict
    q, s = q_int8(W.detach().numpy())
    return {"s": s, "w": b64(q)}

def export_model(net):
    n_params = count_params(net)
    sd = net.state_dict()
    blocks = []
    for i in range(N_LAYERS):
        p = f"blocks.{i}."
        ipw = sd[p + "attn.in_proj_weight"].numpy()           # (3D, D)
        ipb = sd[p + "attn.in_proj_bias"].numpy()              # (3D,)
        blocks.append({
            "ln1": {"g": sd[p + "ln1.weight"].numpy().tolist(), "b": sd[p + "ln1.bias"].numpy().tolist()},
            "qkv": {**qt(torch.from_numpy(ipw)), "b": ipb.tolist()},
            "out": {**qt(sd[p + "attn.out_proj.weight"]), "b": sd[p + "attn.out_proj.bias"].numpy().tolist()},
            "ln2": {"g": sd[p + "ln2.weight"].numpy().tolist(), "b": sd[p + "ln2.bias"].numpy().tolist()},
            "w1": {**qt(sd[p + "fc1.weight"]), "b": sd[p + "fc1.bias"].numpy().tolist()},
            "w2": {**qt(sd[p + "fc2.weight"]), "b": sd[p + "fc2.bias"].numpy().tolist()},
        })
    sal = []
    for g in saliency(net):
        q, s = q_int8(g)
        sal.append({"s": s, "g": b64(q)})
    model = {
        "version": "4.5",
        "params": n_params,
        "hashDim": HASH_DIM, "nFeat": N_FEAT,
        "segs": SEGS, "segDim": SEG_DIM, "dModel": D_MODEL,
        "heads": N_HEADS, "layers": N_LAYERS, "ff": FF,
        "labels": ["Phishing / scam", "Malware / compromise", "Outdated & insecure stack", "Poor quality / SEO"],
        "featNames": ["url_length", "host_length", "host_digit_ratio", "host_hyphens", "subdomains",
            "ip_host", "suspicious_tld", "punycode", "https", "path_depth", "query_length", "at_sign",
            "cred_word_in_url", "has_title", "title_length", "has_description", "description_length",
            "has_viewport", "has_charset", "has_lang", "has_favicon", "h1_count", "img_count",
            "img_no_alt_ratio", "inline_style_count", "script_count", "iframe_count", "form_count",
            "has_password_form", "form_over_http", "mixed_content_count", "inline_event_handlers",
            "has_jquery", "jquery_version", "deprecated_tag_count", "blank_link_noopener_ratio",
            "eval_count", "obfuscation_count", "long_hex_string", "suspicious_keyword_count",
            "crypto_miner_marker", "external_script_count", "http_external_script_ratio",
            "csp_header", "hsts_header", "xfo_header", "meta_refresh", "password_fields"],
        "tokenVocab": {t: fnv1a(t) % HASH_DIM for t in ALL_TOKENS},
        "proj": {**qt(sd["proj.weight"]), "b": sd["proj.bias"].numpy().tolist()},
        "cls": sd["cls"].numpy().tolist(),
        "pos": sd["pos"].numpy()[0].tolist(),
        "blocks": blocks,
        "lnF": {"g": sd["lnF.weight"].numpy().tolist(), "b": sd["lnF.bias"].numpy().tolist()},
        "head": {**qt(sd["head.weight"]), "b": sd["head.bias"].numpy().tolist()},
        "sal": sal,
    }
    with open("model.json", "w") as f:
        json.dump(model, f, separators=(",", ":"))
    print(f"exported model.json  params={n_params:,}", flush=True)

if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    resume = len(sys.argv) > 2 and sys.argv[2] == "resume"
    ckpt = "ckpt.pt"
    net = Net()
    n_params = count_params(net)
    assert n_params == 8_269_972, "unexpected parameter count: %s" % n_params
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=0.01)
    t_start = 0
    if resume and __import__("os").path.exists(ckpt):
        z = torch.load(ckpt, weights_only=False)
        net.load_state_dict(z["model"])
        opt.load_state_dict(z["opt"])
        t_start = z["step"]
        del z
        import gc; gc.collect()
        print(f"resumed at step {t_start}", flush=True)
    net.train()
    train_round(net, opt, steps, t_start)
    t_start += steps
    torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "step": t_start}, ckpt)
    print(f"checkpoint saved at step {t_start}", flush=True)
    if t_start >= TOTAL_STEPS:
        Xv, Yv, pv = validate(net)
        export_model(net)
