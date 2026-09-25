#!/usr/bin/env python3
"""Assemble the template parts and inject the trained model -> index.html."""
import glob, os, sys

# 1) reassemble index-template.html from its parts (shipped as chunks)
parts = sorted(glob.glob("index-template.html.part??"))
if parts and not os.path.exists("index-template.html"):
    with open("index-template.html", "w", encoding="utf-8") as out:
        for p in parts:
            out.write(open(p, encoding="utf-8").read())
    print("assembled index-template.html from", len(parts), "parts")

# 2) inject model.json
tpl, model = "index-template.html", "model.json"
html = open(tpl, encoding="utf-8").read()
if html.count("__MODEL_DATA__") != 1:
    sys.exit("ERROR: expected exactly one __MODEL_DATA__ placeholder in %s" % tpl)
weights = open(model, encoding="utf-8").read()
out = html.replace("__MODEL_DATA__", weights)
open("index.html", "w", encoding="utf-8").write(out)
print("index.html written: {:,} bytes (template {:,} + model {:,})".format(
    len(out), len(html), len(weights)))
