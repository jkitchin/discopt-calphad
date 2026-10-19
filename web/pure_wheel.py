"""Repack an installed package as a pure-Python ``py3-none-any`` wheel.

Used by ``build-web.sh`` for discopt. discopt ships a Rust extension
(``_rust.*.so``) and there is no emscripten build of it, so it cannot be
installed in the browser as published. It does not have to be, *for solving*:
``discopt/modeling/core.py`` imports ``_rust`` lazily, inside the functions that
need the Rust model builder, and equilibrium solving goes through the tape
evaluator into POUNCE instead. Import tracing over a solve confirms it: zero
imports of ``discopt._rust`` -- with ``DISCOPT_ARENA_TAPE=0``, which the worker
sets. From discopt 0.9 the default lowering is the arena tape, which imports
``discopt._rust`` unguarded, and every solve fails without that switch.

Certifying is the exception, and it is not a small one. Global pricing
(``equilibrium(..., certify="final"|"every")``, ``EquilibriumResult.certify()``)
calls ``discopt.Model.solve()``, which imports ``discopt.solver``, which imports
``discopt._rust``. In the browser that raises ``ModuleNotFoundError``. So the
wheel this script builds can solve and can price *locally* -- scipy L-BFGS-B, the
``method="local"`` path, which is what the loop uses and what earns
``STABILITY_SEARCHED`` -- but it can never reach a global certificate. The web
app says so plainly rather than offering a button that throws.

So the extension is dropped and the remaining pure Python is repacked. Two other
edits are needed for micropip to accept the result:

* ``Requires-Dist`` is filtered to what this path actually uses. discopt
  declares jax and jaxlib, which have no wasm build and are never imported here;
  micropip would try to resolve them and fail the install. ``pounce-solver`` is
  dropped for the opposite reason -- it *is* needed, but the emscripten build is
  a local wheel the worker installs first, not something to resolve from PyPI.
* ``Requires-Python`` is relaxed to match this project, since the wheel no
  longer carries a CPython-version-specific binary.

If discopt ever gains a real emscripten wheel, delete this and install that.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import re
import sys
import zipfile
from pathlib import Path

#: Dependencies to keep in the repacked METADATA. Both ship with Pyodide.
KEEP_DEPS = {"numpy", "scipy"}

#: Extensions that cannot survive the trip to wasm.
NATIVE = {".so", ".pyd", ".dylib"}


def repack(site_packages: Path, package: str, out_dir: Path) -> Path:
    src = site_packages / package
    if not src.is_dir():
        raise SystemExit(f"{package} is not installed in {site_packages}")
    dist = next(site_packages.glob(f"{package}-*.dist-info"))
    version = dist.name.removesuffix(".dist-info").split("-", 1)[1]

    records: list[tuple[str, str, int]] = []
    dropped_native: list[str] = []
    dropped_deps: list[str] = []
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:

        def write(arc: str, data: bytes) -> None:
            z.writestr(arc, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            records.append((arc, f"sha256={digest}", len(data)))

        for p in sorted(src.rglob("*")):
            if p.is_dir() or "__pycache__" in p.parts:
                continue
            if p.suffix in NATIVE:
                dropped_native.append(p.name)
                continue
            write(str(p.relative_to(site_packages)), p.read_bytes())

        out_lines = []
        for line in (dist / "METADATA").read_text().splitlines(keepends=True):
            if line.startswith("Requires-Dist:"):
                pkg = re.split(r"[\s\[<>=!;(]", line.split(":", 1)[1].strip())[0].lower()
                if pkg not in KEEP_DEPS:
                    dropped_deps.append(pkg)
                    continue
            if line.startswith("Requires-Python:"):
                line = "Requires-Python: >=3.10\n"
            out_lines.append(line)
        write(f"{dist.name}/METADATA", "".join(out_lines).encode())
        write(
            f"{dist.name}/WHEEL",
            b"Wheel-Version: 1.0\nGenerator: discopt-calphad build-web.sh\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        for extra in ("entry_points.txt", "top_level.txt", "licenses/LICENSE", "LICENSE"):
            p = dist / extra
            if p.is_file():
                write(f"{dist.name}/{extra}", p.read_bytes())

        rec = io.StringIO()
        w = csv.writer(rec, lineterminator="\n")
        for row in records:
            w.writerow(row)
        w.writerow([f"{dist.name}/RECORD", "", ""])
        z.writestr(f"{dist.name}/RECORD", rec.getvalue())

    out_dir.mkdir(parents=True, exist_ok=True)
    wheel = out_dir / f"{package}-{version}-py3-none-any.whl"
    wheel.write_bytes(buf.getvalue())
    print(f"  {wheel.name}  {wheel.stat().st_size / 1e6:.2f} MB")
    print(f"    dropped native: {', '.join(dropped_native) or 'none'}")
    print(f"    dropped deps:   {', '.join(sorted(set(dropped_deps))) or 'none'}")
    return wheel


if __name__ == "__main__":
    repack(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
