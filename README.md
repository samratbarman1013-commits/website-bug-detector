# BugHunter v3.5 — Website Bug & Vulnerability Detector

A **single-file website scanner** that checks any site for real bugs, security
issues and quality problems — and tells you exactly how to fix each one.
Everything runs 100% in the browser: no server, no build step, no data leaving
your machine.

[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/samratbarman1013-commits/website-bug-detector)

Live instance: https://bughunter-website-scanner.netlify.app

## What's new in v3.5

- **AI risk engine — 5,055,484 parameters** (3.84M in v3, 2.37M in v2, 1.5M in v1):
  `2608 → 1440 → 720 → 360 → 4`, retrained and re-verified (AUC ~1.00 on
  all heads, quantized accuracy ≥ 99.6%).
- **Offline chat assistant** — click the 💬 button and ask about your scan,
  any finding, or web security topics (CSP, SRI, HSTS, phishing…). Rule-based,
  runs entirely inside the page, nothing is sent anywhere.
- **UI polish** — page-load animation, refined cards and hover states.
- Everything from v3, v2 and v1 is still included.

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
  JSON report export, scan history, offline chat assistant

**AI layer — 5,055,484 parameters (v3.5)**
- Multi-label neural network: `2560 hashed bag-of-tokens (FNV-1a) + 48 engineered
  features → 1440 → 720 → 360 → 4` sigmoid heads
- Outputs: phishing/scam, malware/compromise, outdated stack, poor quality probabilities
- Trained offline (`train_ai.py`) by distilling expert security heuristics into the
  network on a synthetic corpus, with noise-token and missing-data augmentation
- Exported int8-quantized (per-tensor scale) and embedded in `index.html` — inference
  runs locally in JavaScript on plain typed arrays
- Saliency-based explanations show which tokens/features pushed each verdict

**Offline assistant (v3.5)**
- Rule-based chat: knowledge base of every check the scanner performs plus
  live answers computed from YOUR scan results (priorities, summaries, scores)
- 100% local — no network calls, no data collection, works offline

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
> python3 train_ai.py 330
> python3 train_ai.py 220 resume   # validates + exports model.json
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
- The chat assistant is rule-based (keyword + knowledge base), not a large language
  model. It knows this scanner's checks and your scan results — not general knowledge.
- This is a passive, surface-level scan. It is **not** a penetration test.

## Ethics

Only scan websites you own or have explicit permission to test.

## License

MIT
