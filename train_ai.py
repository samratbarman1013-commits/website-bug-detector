#!/usr/bin/env python3
"""
BugHunter AI risk model — v3.5.
Architecture : MLP 2608 -> 1440 -> 720 -> 360 -> 4 (sigmoid heads)
Inputs       : 2560 hashed bag-of-tokens (FNV-1a % 2560) + 48 numeric features
Outputs      : multi-label threat probabilities
               [phishing/scam, malware/compromise, outdated stack, poor quality]
Parameters   : 5,055,484
Training     : numpy + Adam, BCE loss, on a heuristic-derived synthetic corpus
               (expert rules distilled into a neural net) with noise tokens and
               missing-data augmentation for robustness.
Usage         : python3 train_ai.py <steps_this_round> [resume]
"""
import json, base64, time, sys
import numpy as np

rng = np.random.default_rng(42)

HASH_DIM, N_FEAT = 2560, 48
IN_DIM = HASH_DIM + N_FEAT
H1, H2, H3, NOUT = 1440, 720, 360, 4
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

    # ---------------- class overrides (same order as scalar spec) -------------
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
    # benign tokens: 20-55 per sample (with-replacement approx is fine after dedupe)
    k = rng.integers(20, 55, n)
    draw = rng.integers(0, len(BENIGN), (n, 55))
    mask = rng.random((n, 55)) * 55 < k[:, None]
    BI = VOCAB_IDX["benign"]
    rows, cols = np.nonzero(mask)
    X[rows, BI[draw[rows, cols]]] = 1.0
    # class vocab tokens
    for name, toks, p in VOCAB_IDS:
        if name == "benign":
            continue
        active = tok_counts[name]
        if not active.any():
            continue
        sel = (rng.random((n, len(toks))) < p) & active[:, None]
        r_, c_ = np.nonzero(sel)
        X[r_, VOCAB_IDX[name][c_]] = 1.0
    # noise tokens
    cnt = rng.integers(0, 25, n)
    total = int(cnt.sum())
    if total:
        nidx = rng.integers(0, HASH_DIM, total)
        ridx = np.repeat(np.arange(n), cnt)
        X[ridx, nidx] = 1.0
    X[:, HASH_DIM:] = f
    # missing-page augmentation
    miss = rng.random(n) < 0.03
    X[miss, HASH_DIM + 13: HASH_DIM + N_FEAT] = 0.5

    Y = np.stack([phish, malw, outd, qual], axis=1).astype(np.float32)
    return X, Y

VOCAB_IDS = VOCABS

# ---------------- model -------------------------------------------------------
def init_params():
    def he(out, inn):
        return (rng.standard_normal((out, inn)) * np.sqrt(2.0 / inn)).astype(np.float32)
    return {"W1": he(H1, IN_DIM), "b1": np.zeros(H1, np.float32),
            "W2": he(H2, H1), "b2": np.zeros(H2, np.float32),
            "W3": he(H3, H2), "b3": np.zeros(H3, np.float32),
            "W4": he(NOUT, H3), "b4": np.zeros(NOUT, np.float32)}

def forward(P, X):
    z1 = X @ P["W1"].T + P["b1"]; a1 = np.maximum(z1, 0)
    z2 = a1 @ P["W2"].T + P["b2"]; a2 = np.maximum(z2, 0)
    z3 = a2 @ P["W3"].T + P["b3"]; a3 = np.maximum(z3, 0)
    return z1, a1, z2, a2, z3, a3, a3 @ P["W4"].T + P["b4"]

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def init_opt(P):
    return {k: {"m": np.zeros_like(v), "v": np.zeros_like(v)} for k, v in P.items()}

def adam_step(P, opt, g, step, lr=2e-3, wd=1e-4):
    for k in P:
        g[k] += wd * P[k]
        m, v = opt[k]["m"], opt[k]["v"]
        m *= 0.9; m += 0.1 * g[k]
        v *= 0.999; v += 0.001 * g[k] * g[k]
        mh = m / (1 - 0.9 ** step); vh = v / (1 - 0.999 ** step)
        P[k] -= lr * mh / (np.sqrt(vh) + 1e-8)

def train_round(P, opt, steps, t_start=0, batch=256):
    t0 = time.time()
    for step in range(t_start + 1, t_start + steps + 1):
        X, Y = gen_batch(batch)
        z1, a1, z2, a2, z3, a3, z4 = forward(P, X)
        p = sigmoid(z4)
        n = batch
        dz4 = (p - Y) / n
        g = {}
        g["W4"] = dz4.T @ a3; g["b4"] = dz4.sum(0)
        da3 = dz4 @ P["W4"]; dz3 = da3 * (z3 > 0)
        g["W3"] = dz3.T @ a2; g["b3"] = dz3.sum(0)
        da2 = dz3 @ P["W3"]; dz2 = da2 * (z2 > 0)
        g["W2"] = dz2.T @ a1; g["b2"] = dz2.sum(0)
        da1 = dz2 @ P["W2"]; dz1 = da1 * (z1 > 0)
        g["W1"] = dz1.T @ X; g["b1"] = dz1.sum(0)
        adam_step(P, opt, g, step)
        if step % 50 == 0:
            loss = -np.mean(Y * np.log(p + 1e-9) + (1 - Y) * np.log(1 - p + 1e-9))
            print(f"step {step:4d}  loss {loss:.5f}  ({time.time()-t0:.0f}s)", flush=True)

def auc(y, s):
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0: return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def validate(P, N=20000):
    Xv, Yv = gen_batch(N)
    pv = sigmoid(forward(P, Xv)[-1])
    for c in range(NOUT):
        print(f"val AUC {LABELS[c]:22s} = {auc(Yv[:, c], pv[:, c]):.4f}", flush=True)
    return Xv, Yv, pv

def q_int8(W):
    s = float(np.abs(W).max() / 127.0) or 1e-12
    q = np.clip(np.round(W / s), -127, 127).astype(np.int8)
    return q, s

def b64(arr):
    return base64.b64encode(arr.tobytes()).decode("ascii")

def export_model(P):
    layers = []
    for Wk, bk, inn, out in [("W1", "b1", IN_DIM, H1), ("W2", "b2", H1, H2),
                             ("W3", "b3", H2, H3), ("W4", "b4", H3, NOUT)]:
        q, s = q_int8(P[Wk])
        layers.append({"in": inn, "out": out, "s": s,
                       "b": [float(v) for v in P[bk]], "w": b64(q)})
    sal = []
    for c in range(NOUT):
        g = (P["W1"].T @ P["W2"].T @ P["W3"].T @ P["W4"][c]).astype(np.float32)
        q, s = q_int8(g)
        sal.append({"s": s, "g": b64(q)})
    n_params = IN_DIM * H1 + H1 + H1 * H2 + H2 + H2 * H3 + H3 + H3 * NOUT + NOUT
    model = {
        "version": "3.5",
        "params": n_params,
        "hashDim": HASH_DIM, "nFeat": N_FEAT,
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
        "layers": layers, "sal": sal,
    }
    with open("model.json", "w") as f:
        json.dump(model, f, separators=(",", ":"))
    print(f"exported model.json  params={n_params:,}", flush=True)

if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    resume = len(sys.argv) > 2 and sys.argv[2] == "resume"
    ckpt = "ckpt.npz"
    if resume:
        z = np.load(ckpt, allow_pickle=True)
        P = {k: z[k] for k in z.files if "|" not in k and k != "step"}
        opt = {}
        for k in z.files:
            if k.endswith("|m"):
                base = k[:-2]
                opt[base] = {"m": z[k], "v": z[base + "|v"]}
        t_start = int(z["step"])
        print(f"resumed at step {t_start}", flush=True)
    else:
        P = init_params(); opt = init_opt(P); t_start = 0
    train_round(P, opt, steps, t_start)
    np.savez(ckpt, step=t_start + steps, **P,
             **{f"{k}|m": opt[k]["m"] for k in P}, **{f"{k}|v": opt[k]["v"] for k in P})
    print(f"checkpoint saved at step {t_start + steps}", flush=True)
    if (t_start + steps) >= 550:
        validate(P)
        export_model(P)
