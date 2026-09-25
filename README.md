# BugHunter v4.5 — Website Bug & Vulnerability Detector

A **single-file website scanner** that checks any site for real bugs, security
issues and quality problems — and tells you exactly how to fix each one.
The scanner, the AI risk transformer and the Thoughts view run 100% in the
browser; the chat assistant can additionally use an online large language
model for real, general answers (with an offline knowledge-base fallback).

[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/samratbarman1013-commits/website-bug-detector)

Live instance: https://bughunter-website-scanner.netlify.app

## What's new in v4.5

- **REAL transformer — 8,269,972 parameters** (7.02M MLP in v4, 5.06M in v3.5,
  3.84M in v3, 2.37M in v2, 1.5M in v1). The risk model is now a genuine
  pre-LN transformer encoder: 8 input segments → 336-dim token embeddings,
  6 layers of multi-head self-attention (8 heads) with 336→1344→336
  feed-forward blocks, a CLS token readout and 4 sigmoid heads. Trained from
  scratch with PyTorch (AdamW, BCE), verified: AUC 1.00 on all four heads,
  quantized accuracy 98.9–99.98%, and the in-browser JavaScript forward pass
  reproduces the Python inference to 7 decimal places.
- **Real AI chat** — the assistant now answers through a large language model
  (online) with your scan results injected as context, so you can ask
  anything, not just canned topics. If the AI service is unreachable (or you
  click the header label to go offline), it falls back to an expanded
  built-in knowledge base (XSS, SQL injection, CSRF, TLS, CORS, cookies,
  2FA, backups, CSP, SRI, HSTS, phishing…).
- **Richer Thoughts** — the live reasoning trace now names the actual findings
  and the detected technology stack.
- Everything from v4 and earlier is still included (Thoughts view, 30+
  checks, scan history, saliency explanations, JSON reports).

## What it detects

**Rule-based engine**
- Reachability & HTTPS/TLS (direct network probes)
- Security headers (CSP, HSTS, X-Frame-Options, X-Content-Type-Options, Referrer-Policy)
- Sensitive file exposure — `/.env`, `/.git/config`, `/.DS_Store` with content-signature verification
- Outdated libraries (jQuery < 3.x with known CVEs), mixed content, reverse tabnabbing
- Password forms over plain HTTP, crypto-miner markers
- SEO & HTML quality: title, meta description, viewport, charset, lang, alt text,
  deprecated tags, heading structure, robots.txt / sitemap.xml
- v2: Subresource Integrity (SRI), unsandboxed iframes, `javascript:` links,
  duplicate DOM IDs, image width/height (CLS), unlabeled form inputs,
  meta-refresh redirects, inline-handler audit
- v3: layout tables, missing canonical URL, autoplaying media,
  generic link text, empty links/buttons, password-field autocomplete hints
- Technology detection (WordPress, React, Laravel, Cloudflare, …)
- 0–100 score, severity-grouped findings, fix + copy-paste code for every issue,
  JSON report export, scan history, live Thoughts view

**AI layer — 8,269,972-parameter transformer (v4.5)**
- Pre-LN transformer encoder: 2,560 hashed bag-of-tokens (FNV-1a) + 48
  engineered features → 8 segments of 326 → token projection to 336 dims →
  6 self-attention layers (8 heads) with 336→1344→336 FFN → CLS readout →
  4 sigmoid heads
- Outputs: phishing/scam, malware/compromise, outdated stack, poor quality probabilities
- Trained offline (`train_ai.py`, PyTorch CPU) by distilling expert security
  heuristics into the network on a synthetic corpus, with noise-token and
  missing-data augmentation
- Exported int8-quantized (per-tensor scale) and embedded in `index.html` —
  the same transformer forward pass (attention, LayerNorm, FFN) is
  re-implemented in JavaScript on typed arrays, parity-checked against Python
- Saliency-based explanations show which tokens/features pushed each verdict

**Chat assistant (v4.5)**
- Online mode (default): your question + a short scan summary are sent to a
  third-party AI service (text.pollinations.ai) so a large language model can
  answer anything — general questions included
- Offline mode (click the label in the chat header): rule-based knowledge
  base + live answers computed from YOUR scan results, nothing sent anywhere
- Automatic fallback to offline mode when the AI service is unreachable

**Thoughts view (v4)**
- A live reasoning trace rendered from the scanner's real intermediate state:
  plan, TLS verdict, source fetch, rule-engine summary (with the actual
  finding names), transformer verdict with top signals, sensitive-file
  probes, and the final score weighing
- Real data only — every thought is generated from actual scan results as
  they happen, not a canned script

## Usage

Open `index.html` in any modern browser, type a domain, hit **RUN SCAN**.
Or deploy it anywhere static (Netlify, GitHub Pages, …) — it is one file.

> `index.html` and `model.json` are **built artifacts**: GitHub Actions
> (`.github/workflows/build.yml`) retrains the transformer deterministically,
> verifies the quantized accuracy, and commits them automatically whenever the
> sources change. To rebuild locally:
>
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install numpy
> python3 train_ai.py 160
> python3 train_ai.py 90 resume   # validates + exports model.json (250 total)
> python3 verify_model.py
> python3 build.py                # reassembles template parts + injects model
> ```

## Files

| File | Purpose |
|---|---|
| `index.html` | Complete scanner + embedded quantized transformer (CI-built) |
| `index-template.html.part??` | Scanner source (template), shipped in 3 chunks |
| `train_ai.py` | Transformer training / int8 export script (PyTorch) |
| `build.py` | Reassembles template, injects `model.json` → `index.html` |
| `verify_model.py` | Quantized-transformer accuracy gate (numpy re-implementation) |
| `model.json` | Exported int8-quantized transformer weights (CI-built) |
| `netlify.toml` | Static-publish config for Netlify |

## Honest limitations

- Browser CORS limits: page HTML is fetched directly when possible, otherwise via
  public CORS proxies (corsproxy.io, AllOrigins, codetabs). Response headers are
  only readable when the target site sends CORS headers.
- The transformer is trained on heuristic-derived synthetic data — it is a
  distillation of expert rules, not a model trained on a real phishing/malware
  corpus. Treat its scores as a lead, not proof.
- The chat's online mode depends on a free third-party AI service; it can be
  slow or unavailable, in which case the offline knowledge base answers instead.
- The Thoughts view renders the scanner's actual reasoning steps from live scan
  data (it is a trace of real checks and real AI outputs, not a separate model).
- Transformer inference runs in JavaScript on the CPU (~0.5–1 s per scan,
  once per scan).
- This is a passive, surface-level scan. It is **not** a penetration test.

## Ethics

Only scan websites you own or have explicit permission to test.

## License

MIT
