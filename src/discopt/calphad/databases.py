"""Published TDB databases, looked up by name and downloaded on demand.

This package does not ship thermodynamic databases. It ships a registry
(``databases.json``). Each entry names one published TDB file and gives:

- a URL pinned to a fixed commit, and the file's SHA-256;
- the assessment it comes from, and where the file is published;
- the elements it covers;
- ``binaries``: for each binary system the file can form, which of its phases
  this package cannot model yet.

:func:`fetch` returns a local path. It tries three places in turn:

1. the user cache;
2. a local mirror, for example the copy pycalphad installs with its tests;
3. the URL. A download goes into the cache.

A file is used only if its hash matches, so a pinned name always means the same
parameters.

    >>> from discopt.calphad import databases
    >>> [e.name for e in databases.find(["AL", "ZN"], complete=True)]
    ['alzn_mey']
    >>> db = databases.load("alzn_mey")            # doctest: +SKIP

Set ``DISCOPT_CALPHAD_CACHE`` to change where downloads are kept.

Where to find more TDB files: TDBDB (https://avdwgroup.engin.brown.edu/)
indexes the files published with the CALPHAD literature. NIMS CPDDB
(https://cpddb.nims.go.jp/) has several hundred assessments, and needs a free
login. NIST publishes assessments in its Materials Data Repository. A file from
any of these can be read directly with
:meth:`~discopt.calphad.ThermoDatabase.from_tdb`. Add it to the registry when it
has a stable URL.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import urllib.request
from dataclasses import dataclass, field
from functools import cache
from itertools import combinations
from pathlib import Path

__all__ = ["Entry", "cache_dir", "describe", "fetch", "find", "get", "load", "registry"]

REGISTRY_FILE = Path(__file__).with_name("databases.json")


@dataclass(frozen=True)
class Entry:
    """One published TDB file in the registry."""

    name: str
    title: str
    #: the upstream file name
    file: str
    url: str
    sha256: str
    size: int
    #: the assessment the parameters come from
    reference: str
    #: where the file itself is published, and on what terms
    source: str
    elements: tuple[str, ...]
    #: ``"A-B"`` -> phases of that binary this package cannot model yet; empty when every phase is supported
    binaries: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: local copies to try before downloading, as ``"package:relative/path"``
    mirrors: tuple[str, ...] = ()
    notes: str = ""

    def pairs(self, complete: bool = False) -> list[str]:
        """The binary systems this file can form, or only those with every phase supported."""
        return [k for k, missing in self.binaries.items() if not (complete and missing)]

    def unsupported(self, elements) -> tuple[str, ...] | None:
        """Phases of this binary that cannot be modeled yet, or None if the file does not form it."""
        return self.binaries.get(pair_key(elements))


def pair_key(elements) -> str:
    """The registry's key for a binary: the two symbols, upper case, sorted, joined by ``-``."""
    return "-".join(sorted(e.upper() for e in elements))


@cache
def registry() -> dict[str, Entry]:
    """Every registered database, by name, in registry order."""
    raw = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    out = {}
    for item in raw["databases"]:
        item = dict(item)
        item["elements"] = tuple(item["elements"])
        item["binaries"] = {k: tuple(v) for k, v in item.get("binaries", {}).items()}
        item["mirrors"] = tuple(item.get("mirrors", ()))
        out[item["name"]] = Entry(**item)
    return out


def get(name: str) -> Entry:
    """The registry entry called ``name``."""
    try:
        return registry()[name]
    except KeyError:
        known = ", ".join(registry())
        raise KeyError(f"no database {name!r} in the registry; known: {known}") from None


def find(elements, *, complete: bool = False) -> list[Entry]:
    """Registered databases that cover every element in ``elements``.

    For a binary, ``complete=True`` keeps only the files in which every phase
    of that system can be modeled. The order is: complete files first, then the
    files with the fewest elements. A dedicated binary assessment therefore comes
    before a multicomponent database that also covers the pair.
    """
    want = {e.upper() for e in elements}
    hits = [e for e in registry().values() if want <= set(e.elements)]
    if len(want) == 2:
        hits = [e for e in hits if e.unsupported(want) is not None]
        if complete:
            hits = [e for e in hits if not e.unsupported(want)]
        return sorted(hits, key=lambda e: (bool(e.unsupported(want)), len(e.elements)))
    return sorted(hits, key=lambda e: len(e.elements))


def cache_dir() -> Path:
    """Where downloaded databases are kept.

    ``$DISCOPT_CALPHAD_CACHE`` if it is set. Otherwise the platform's user cache:
    ``~/Library/Caches`` on macOS, ``%LOCALAPPDATA%`` on Windows, and
    ``$XDG_CACHE_HOME`` or ``~/.cache`` elsewhere. In each case the files go in a
    ``discopt-calphad/tdb`` subdirectory.
    """
    env = os.environ.get("DISCOPT_CALPHAD_CACHE")
    if env:
        return Path(env).expanduser()
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "discopt-calphad" / "tdb"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mirror_path(mirror: str) -> Path | None:
    """Resolve ``"package:relative/path"`` without importing the package."""
    pkg, _, rel = mirror.partition(":")
    try:
        spec = importlib.util.find_spec(pkg)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    for root in spec.submodule_search_locations:
        path = Path(root) / rel
        if path.is_file():
            return path
    return None


def fetch(name: str, *, refresh: bool = False, timeout: float = 60.0) -> Path:
    """A local path to the registered database ``name``, verified against its SHA-256.

    The user cache is tried first, then the entry's local mirrors, then its URL.
    A download goes into the cache. ``refresh=True`` downloads again even when a
    good copy exists. A file whose hash does not match is never returned: a
    mismatched download raises ``ValueError`` and is not kept.
    """
    entry = get(name)
    cached = cache_dir() / f"{entry.name}.tdb"
    if not refresh:
        if cached.is_file() and _sha256(cached) == entry.sha256:
            return cached
        for mirror in entry.mirrors:
            path = _mirror_path(mirror)
            if path is not None and _sha256(path) == entry.sha256:
                return path

    request = urllib.request.Request(entry.url, headers={"User-Agent": "discopt-calphad"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
    except OSError as err:
        raise OSError(f"could not download {entry.name} from {entry.url}: {err}") from err
    digest = hashlib.sha256(data).hexdigest()
    if digest != entry.sha256:
        raise ValueError(
            f"{entry.name}: the file at {entry.url} has SHA-256 {digest}, but the registry pins "
            f"{entry.sha256}. It was not saved."
        )
    cached.parent.mkdir(parents=True, exist_ok=True)
    # Write, then rename into place. A reader never sees a partial file, and two
    # processes downloading the same file at once do not corrupt each other.
    with tempfile.NamedTemporaryFile(dir=cached.parent, delete=False, suffix=".part") as tmp:
        tmp.write(data)
    os.replace(tmp.name, cached)
    return cached


def load(name: str, **kwargs):
    """Fetch the registered database ``name`` and read it into a :class:`~discopt.calphad.ThermoDatabase`.

    Keyword arguments are passed on to
    :meth:`~discopt.calphad.ThermoDatabase.from_tdb`.
    """
    from discopt.calphad.database import ThermoDatabase

    return ThermoDatabase.from_tdb(fetch(name), **kwargs)


def describe(db) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """The elements of a database, and its unsupported phases in each binary.

    ``db`` is a :class:`~discopt.calphad.ThermoDatabase` or a path to a TDB
    file. This computes the ``elements`` and ``binaries`` fields of a registry
    entry. A binary is listed when at least two of its phases can be modeled;
    with fewer there is nothing to draw.
    """
    from discopt.calphad.database import ThermoDatabase

    if not isinstance(db, ThermoDatabase):
        db = ThermoDatabase.from_tdb(db)
    elements = tuple(sorted(e for e in db.elements if e not in ("VA", "/-")))
    binaries = {}
    for pair in combinations(elements, 2):
        status = db.applicability(pair)
        ok = sum(v == "ok" for v in status.values())
        if ok >= 2:
            missing = (k for k, v in status.items() if v.startswith("UnsupportedModelError"))
            binaries[pair_key(pair)] = tuple(sorted(missing))
    return elements, binaries
