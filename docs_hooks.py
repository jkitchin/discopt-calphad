"""MkDocs hook: copy the logo and the stored example figures into the docs tree.

MkDocs only serves files under ``docs/``, but the logo lives in ``images/`` and
the example figures beside their scripts in ``examples/``. Files are copied only
when changed, so ``mkdocs serve`` does not rebuild in a loop.
"""

import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parent


def _sync(src: pathlib.Path, dest: pathlib.Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.read_bytes() != src.read_bytes():
        shutil.copyfile(src, dest)


def on_pre_build(config, **kwargs):
    _sync(ROOT / "images" / "logo.png", ROOT / "docs" / "assets" / "logo.png")
    for png in sorted((ROOT / "examples").glob("*.png")):
        _sync(png, ROOT / "docs" / "assets" / "examples" / png.name)
