"""The TDB registry: its contents, lookup, and fetching with hash checks (no network)."""

import hashlib
import io
import re
import urllib.request

import pytest

from discopt.calphad import databases
from discopt.calphad.databases import fetch, find, get, registry

ENTRIES = list(registry().values())


def test_registry_is_well_formed():
    assert len({e.name for e in ENTRIES}) == len(ENTRIES)
    for e in ENTRIES:
        # pinned to a commit, not a branch or tag, so the hash cannot drift
        assert re.search(r"/[0-9a-f]{40}/", e.url), e.name
        assert re.fullmatch(r"[0-9a-f]{64}", e.sha256), e.name
        assert e.reference and e.source and e.elements
        assert list(e.elements) == sorted(e.elements)
        for key in e.binaries:
            assert set(key.split("-")) <= set(e.elements), (e.name, key)


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e.name)
def test_registry_matches_the_file(entry):
    """The committed hash, elements and support status agree with the file (see tools/refresh_registry.py)."""
    pytest.importorskip("pycalphad")  # every entry is mirrored by pycalphad's test databases
    path = fetch(entry.name)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry.sha256
    elements, binaries = databases.describe(path)
    assert elements == entry.elements
    assert binaries == entry.binaries, "support changed: run tools/refresh_registry.py"


def test_find_prefers_complete_then_dedicated():
    names = [e.name for e in find(["zn", "al"])]
    assert names[0] == "alzn_mey"  # complete and binary, ahead of COST507
    assert "cost507" in names
    assert [e.name for e in find(["AL", "ZN"], complete=True)] == ["alzn_mey"]
    assert find(["AL", "FE"], complete=True) == []
    assert all(e.unsupported(["AL", "FE"]) for e in find(["AL", "FE"]))
    assert find(["XX", "YY"]) == []


def test_get_unknown_name_lists_known():
    with pytest.raises(KeyError, match="alzn_mey"):
        get("nope")


# -- fetching, with the network replaced ------------------------------------

BODY = b"$ a tiny TDB\nELEMENT AL FCC_A1 0 0 0 !\n"


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """A registry of one entry whose URL serves BODY, a cache in tmp_path, and a download counter."""
    entry = databases.Entry(
        name="tiny",
        title="tiny",
        file="tiny.tdb",
        url="https://example.invalid/0000000000000000000000000000000000000000/tiny.tdb",
        sha256=hashlib.sha256(BODY).hexdigest(),
        size=len(BODY),
        reference="none",
        source="test",
        elements=("AL",),
    )
    monkeypatch.setattr(databases, "registry", lambda: {"tiny": entry})
    monkeypatch.setenv("DISCOPT_CALPHAD_CACHE", str(tmp_path / "cache"))
    served = {"body": BODY, "count": 0}

    def urlopen(request, timeout):
        assert request.full_url == entry.url
        served["count"] += 1
        return io.BytesIO(served["body"])

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return served


def test_fetch_downloads_once_into_the_cache(fake, tmp_path):
    path = fetch("tiny")
    assert path == tmp_path / "cache" / "tiny.tdb"
    assert path.read_bytes() == BODY and fake["count"] == 1
    assert fetch("tiny") == path and fake["count"] == 1  # a cache hit
    fetch("tiny", refresh=True)
    assert fake["count"] == 2


def test_fetch_rejects_a_changed_file(fake, tmp_path):
    fake["body"] = BODY + b"$ edited upstream\n"
    with pytest.raises(ValueError, match="SHA-256"):
        fetch("tiny")
    assert not list((tmp_path / "cache").glob("*"))  # nothing kept


def test_fetch_replaces_a_corrupt_cached_copy(fake, tmp_path):
    cached = tmp_path / "cache" / "tiny.tdb"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"truncated")
    assert fetch("tiny").read_bytes() == BODY and fake["count"] == 1


def test_cache_dir_honours_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("DISCOPT_CALPHAD_CACHE", str(tmp_path))
    assert databases.cache_dir() == tmp_path
