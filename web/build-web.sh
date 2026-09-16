#!/usr/bin/env bash
# Stage everything the page needs that is not source: three wheels, the
# thermodynamic databases, and a manifest naming them.
#
# Nothing staged here is committed (see .gitignore). Two of the three wheels are
# built from this checkout, so they would go stale the moment the source moved.
#
# The third -- the emscripten build of pounce-solver -- is different: it is a
# release artifact of the POUNCE repo, pinned to one Pyodide version, and it is
# not on PyPI in that form, so no amount of installing can reproduce it here or
# in CI. That one is committed, in web/vendor/, which is why a fresh clone can
# build this page without a POUNCE checkout. $POUNCE_WHEEL overrides it; a
# POUNCE checkout ($POUNCE_REPO) is the fallback when vendor/ is empty.
#
#   ./web/build-web.sh          # stage, then: python3 -m http.server -d web
#   ./web/build-web.sh --check  # verify what is staged matches this script
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"
wheels="$here/wheels"
data="$here/data"
# Committed, and deliberately not under $wheels: the staging step below starts
# with `rm -rf "$wheels"`, which would delete the one wheel we cannot rebuild.
vendor="$here/vendor"

# Keep in step with worker.js: an emscripten wheel is valid for exactly one
# Pyodide build, and the worker refuses to install one built for another.
PYODIDE_VERSION=0.28.3
POUNCE_REPO="${POUNCE_REPO:-$HOME/projects/pounce}"

page_pyodide() {
  sed -n "s/^const PYODIDE_VERSION = '\(.*\)';$/\1/p" "$here/worker.js"
}

if [[ "${1:-}" == "--check" ]]; then
  page="$(page_pyodide)"
  if [[ "$page" != "$PYODIDE_VERSION" ]]; then
    echo "worker.js pins Pyodide $page but this script stages for $PYODIDE_VERSION" >&2
    exit 1
  fi
  for f in "$wheels/manifest.json" "$data/alzn_mey.tdb" "$data/cumg_nims.tdb"; do
    [[ -f "$f" ]] || { echo "not staged: $f (run web/build-web.sh)" >&2; exit 1; }
  done
  ls "$vendor"/pounce_solver-*-emscripten_*_wasm32.whl >/dev/null 2>&1 ||
    { echo "missing the committed emscripten wheel in $vendor" >&2; exit 1; }
  echo "staged artifacts look current (Pyodide $PYODIDE_VERSION)"
  exit 0
fi

rm -rf "$wheels" "$data"
mkdir -p "$wheels" "$data"

echo "==> discopt, repacked without its Rust extension"
site_packages="$(cd "$root" && uv run python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
(cd "$root" && uv run python "$here/pure_wheel.py" "$site_packages" discopt "$wheels")

echo "==> discopt-calphad"
(cd "$root" && uv build --wheel --out-dir "$wheels" >/dev/null)
echo "  $(cd "$wheels" && ls discopt_calphad-*.whl)"

echo "==> pounce-solver, built for emscripten"
pounce_src="${POUNCE_WHEEL:-}"
if [[ -z "$pounce_src" ]]; then
  pounce_src="$(ls "$vendor"/pounce_solver-*-emscripten_*_wasm32.whl 2>/dev/null | head -1 || true)"
fi
if [[ -z "$pounce_src" ]]; then
  pounce_src="$(ls "$POUNCE_REPO"/crates/pounce-wasm/web-python/wheels/pounce_solver-*-emscripten_*_wasm32.whl 2>/dev/null | head -1 || true)"
fi
if [[ -z "$pounce_src" || ! -f "$pounce_src" ]]; then
  cat >&2 <<EOF
no emscripten pounce-solver wheel found.

One is normally committed in web/vendor/. If it is missing: it is not on PyPI -- it is
built by the POUNCE repo's crates/pounce-wasm/build-wheel.sh and shipped with its
browser page. Point at a checkout with POUNCE_REPO=<path>, or at the wheel itself with
POUNCE_WHEEL=<path>.
EOF
  exit 1
fi
cp "$pounce_src" "$wheels/"
echo "  $(basename "$pounce_src")"

echo "==> databases"
for tdb in "$root/examples/data/alzn_mey.tdb" "$root/examples/data/cumg_nims.tdb"; do
  cp "$tdb" "$data/"
  echo "  $(basename "$tdb")"
done

echo "==> manifest"
python3 - "$wheels/manifest.json" "$PYODIDE_VERSION" "$(basename "$pounce_src")" \
  "$(cd "$wheels" && ls discopt-*.whl)" "$(cd "$wheels" && ls discopt_calphad-*.whl)" <<'PY'
import json
import re
import sys

out, pyodide, pounce, discopt, calphad = sys.argv[1:6]
# The ABI tag carries the emscripten version Pyodide was built with. The worker
# reports a mismatch against this rather than letting micropip call it "no
# matching distribution", which reads like a missing package.
m = re.search(r"emscripten_(\d+)_(\d+)_(\d+)_wasm32", pounce)
json.dump(
    {
        "pyodide_version": pyodide,
        "emscripten_version": ".".join(m.groups()) if m else None,
        "wheels": {"pounce": pounce, "discopt": discopt, "discopt_calphad": calphad},
    },
    open(out, "w"),
    indent=2,
)
print(f"  {out}")
PY

echo
echo "staged. serve it with:  python3 -m http.server -d $here 8000"
