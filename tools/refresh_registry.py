"""Fill in the computed fields of src/discopt/calphad/databases.json.

A person writes the fields that need judgment: name, title, file, url,
reference, source, mirrors and notes. This script downloads each file and
computes the rest from it: sha256, size, elements and binaries. Run it after
adding an entry, or when phase-model support changes. The binaries field says
which phases cannot be modeled yet, so it changes when support is added.

    uv run python tools/refresh_registry.py            # rewrite databases.json
    uv run python tools/refresh_registry.py --offline  # use local mirrors only, no network

tests/test_databases.py checks that the committed file agrees with what this
script would write.
"""

import argparse
import hashlib
import json
import tempfile
import urllib.request
import warnings
from pathlib import Path

from discopt.calphad import databases

FIELDS = ["name", "title", "file", "url", "sha256", "size", "reference", "source", "mirrors", "notes", "elements"]


def content(item, offline):
    if offline:
        for mirror in item.get("mirrors", []):
            path = databases._mirror_path(mirror)
            if path is not None:
                return path.read_bytes()
        raise SystemExit(f"{item['name']}: no local mirror found, and --offline was given")
    req = urllib.request.Request(item["url"], headers={"User-Agent": "discopt-calphad"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def dump(doc):
    """JSON, indented, with each binary on one line so COST507's hundreds of pairs stay readable."""
    j = lambda v: json.dumps(v, ensure_ascii=False)
    lines = ["{"]
    lines += [f"  {j(k)}: {j(v)}," for k, v in doc.items() if k != "databases"]
    lines.append('  "databases": [')
    for n, item in enumerate(doc["databases"]):
        lines.append("    {")
        for k, v in item.items():
            if k != "binaries":
                lines.append(f"      {j(k)}: {j(v)},")
        pairs = [f"        {j(k)}: {j(v)}" for k, v in item["binaries"].items()]
        lines.append('      "binaries": {' + ("\n" + ",\n".join(pairs) + "\n      }" if pairs else "}"))
        lines.append("    }" + ("," if n < len(doc["databases"]) - 1 else ""))
    lines += ["  ]", "}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="read each file from its local mirror")
    args = ap.parse_args()

    path = databases.REGISTRY_FILE
    doc = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for item in doc["databases"]:
        data = content(item, args.offline)
        with tempfile.NamedTemporaryFile(suffix=".tdb", delete=False) as tmp:
            tmp.write(data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            elements, binaries = databases.describe(tmp.name)
        Path(tmp.name).unlink()
        item.update(
            sha256=hashlib.sha256(data).hexdigest(),
            size=len(data),
            elements=list(elements),
        )
        ordered = {k: item[k] for k in FIELDS if k in item}
        ordered["binaries"] = {k: list(v) for k, v in binaries.items()}
        out.append(ordered)
        complete = sum(not v for v in binaries.values())
        print(f"{item['name']:18s} {len(elements):2d} elements, {complete}/{len(binaries)} binaries complete")
    doc["databases"] = out
    path.write_text(dump(doc), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
