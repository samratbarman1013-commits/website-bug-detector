# BugHunter v3 — Website Bug & Vulnerability Detector

A **single-file website scanner** that checks any site for real bugs, security
issues and quality problems — and tells you exactly how to fix each one.
Everything runs 100% in the browser: no server, no build step, no data leaving
your machine.

[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/samratbarman1013-commits/website-bug-detector)

Live instance: https://bughunter-website-scanner.netlify.app

## What's new in v3

- **AI risk engine — 3,837,028 parameters** (2.37M in v2, 1.5M in v1):
  `2608 → 1152 → 576 → 288 → 4`, retrained and re-verified (AUC ~1.00 on
  all heads, quantized accuracy ≥ 99.7%).
- **6 new checks** — layout tables, missing canonical URL, autoplaying media,
  generic "click here" link text, empty links/buttons with no accessible name,
  password-field autocomplete hints.
- Everything from v2 (SRI, sandboxed iframes, CLS, form labels, scan history)
  and v1 is still included.

## What it detects

**Rule-based engine**
- Reachability & HTTPS/TLS (direct network probes)
- Security headers (CSP, HSTS, X-Frame-Options, X-Content-Type-Options, Referrer-Policy)
- Sensitive file exposure — `/.env`, `/.git/config`, `/.DS_Store` with content-signature verification
- Outdated libraries (jQuery < 3.x with known CVEs), mixed content, reverse tabnabbing
- Password forms over plain HTTP, crypto-miner markers
- SEO & HTML quality: title, meta description, viewport, charset, lang, alt text,
  deprecated tags, heading structure, robots.txt / sitemap.xml
- **v2:** Subresource Integrity (SRI), unsandboxed iframes, `javascript:` links,
  duplicate DOM IDs, image width/height (CLS), unlabeled form inputs,
  meta-refresh redirects, inline-handler audit
- **v3:** layout tables, missing canonical URL, autoplaying media,
  generic link text, empty links/buttons, password-field autocomplete hints
- Technology detection (WordPress, React, Laravel, Cloudflare, …)
- 0–100 score, severity-grouped findings, fix + copy-paste code for every issue,
  JSON report export, scan history

**AI layer — 3,837,028 parameters (v3)**
- Multi-label neural network: `2560 hashed bag-of-tokens (FNV-1a) + 48 engineered
  features → 1152 → 576 → 288 → 4` sigmoid heads
- Outputs: phishing/scam, malware/compromise, outdated stack, poor quality probabilities
- Trained offline (`train_ai.py`) by distilling expert security heuristics into the
  network on a synthetic corpus, with noise-token and missing-data augmentation
- Exported int8-quantized (per-tensor scale) and embedded in `index.html` — inference
  runs locally in JavaScript on plain typed arrays
- Saliency-based explanations show which tokens/features pushed each verdict

## Usage

Open `index.html` in any modern browser, type a domain, hit **RUN SCAN**.
Or deploy it anywhere static (Netlify, GitHub Pages, …) — it is one file.

> `index.html` and `model.json` are **built artifacts**: GitHub Actions
> (`.github/workflows/build.yml`) retrains the model deterministically, verifies
> the quantized accuracy, and commits them automatically whenever the sources change.
> To rebuild locally:
>
> ```bash
> pip install numpy
> python3 train_ai.py 270
> python3 train_ai.py 280 resume   # validates + exports model.json
> python3 verify_model.py
> python3 build.py                # reassembles template parts + injects model
> ```

## Files

| File | Purpose |
|---|---|
| `index.html` | Complete scanner + embedded quantized model (CI-built) |
| `index-template.html.part??` | Scanner source (template), shipped in 3 chunks |
| `train_ai.py` | Model training / int8 export script (numpy only) |
| `build.py` | Reassembles template, injects `model.json` → `index.html` |
| `verify_model.py` | Quantized-model accuracy gate |
| `model.json` | Exported int8-quantized weights (CI-built) |
| `netlify.toml` | Static-publish config for Netlify |

## Honest limitations

- Browser CORS limits: page HTML is fetched directly when possible, otherwise via
  public CORS proxies (corsproxy.io, AllOrigins, codetabs). Response headers are
  only readable when the target site sends CORS headers.
- The AI model was trained on heuristic-derived synthetic data — it is a distillation
  of expert rules, not a model trained on a real phishing/malware corpus. Treat its
  scores as a lead, not proof.
- This is a passive, surface-level scan. It is **not** a penetration test.

## Ethics

Only scan websites you own or have explicit permission to test.

## License

MIT
