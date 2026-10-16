"""TDB reader and writer.

Commands handled: ELEMENT, SPECIES, FUNCTION, PHASE, CONSTITUENT, PARAMETER
and TYPE_DEFINITION (stored raw). Bookkeeping commands such as
DEFINE_SYSTEM_DEFAULT, DEFAULT_COMMAND and the reference commands are ignored.
Any other command raises :class:`TDBSyntaxError`, never a silent skip. Command
keywords may be abbreviated per underscore-separated part (``PARA``,
``TYPE_DEF``). Text from ``$`` to the end of a line is a comment.
"""

from __future__ import annotations

import re

from discopt.calphad.symbolic import TDBSyntaxError, parse_piecewise

__all__ = ["parse_tdb", "read_tdb", "write_tdb"]

_HANDLED = ("ELEMENT", "SPECIES", "FUNCTION", "PARAMETER", "PHASE", "CONSTITUENT", "TYPE_DEFINITION")
_IGNORED = (
    "DEFINE_SYSTEM_DEFAULT",
    "DEFAULT_COMMAND",
    "DATABASE_INFO",
    "VERSION_DATE",
    "VERSION_DATA",
    "REFERENCE_FILE",
    "ADD_REFERENCES",
    "LIST_OF_REFERENCES",
    "ASSESSED_SYSTEMS",
    "TEMPERATURE_LIMITS",
)
_KEYWORDS = _HANDLED + _IGNORED


def _keyword(token: str) -> str:
    tok = token.upper().replace("-", "_")  # 'TEMP-LIM' is written for TEMPERATURE_LIMITS
    if tok in _KEYWORDS:
        return tok
    parts = tok.split("_")
    hits = [
        k
        for k in _KEYWORDS
        if len(parts) <= len(k.split("_")) and all(kp.startswith(tp) for tp, kp in zip(parts, k.split("_")))
    ]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise TDBSyntaxError(f"unsupported TDB command {token!r}")
    raise TDBSyntaxError(f"ambiguous TDB command abbreviation {token!r}: {hits}")


def _split_name(name: str) -> tuple[str, tuple[str, ...]]:
    base, _, hint = name.upper().partition(":")
    return base, ((hint,) if hint else ())


def _constituent_list(text: str) -> tuple[str, ...]:
    return tuple(c.strip().rstrip("%").upper() for c in text.replace(",", " ").split() if c.strip())


def _num(text: str) -> float:
    """One numeric field, tolerating FORTRAN's dropped exponent letter.

    Published files sometimes write a fixed-width number with the ``E`` left
    out: COST507 gives yttrium's mass as ``8.89059+01``, which ``float``
    rejects, and one such field is enough to lose a 27-element database. The
    repair is deliberately confined to fields holding a single number, because
    inside an expression those same characters are addition and ``116000+58*T``
    has to keep meaning what it says.
    """
    try:
        return float(text)
    except ValueError:
        return float(re.sub(r"(?<=[\d.])([+-])(\d+)$", r"E\1\2", text.strip()))


def parse_tdb(text: str, db=None, *, skipped: list | None = None):
    """Parse TDB text into ``db`` (a new :class:`ThermoDatabase` if None) and return it.

    Given a list for ``skipped``, a command that will not parse is appended to it
    as ``{"command", "reason"}`` and parsing continues; by default one bad command
    raises and the whole file is lost. COST507 is why: it has a ``!`` typed where
    a ``1`` belongs, and because ``!`` ends a command that one keystroke both
    truncates a PARAMETER and leaves the rest of its number as an orphan command
    -- two casualties that would otherwise cost the other 2,584. Reporting is not
    optional, though: a dropped G parameter is wrong energies rather than a
    missing feature, so the caller is handed every command it lost.
    """
    from discopt.calphad.database import Element, Parameter, Phase, Species, ThermoDatabase

    db = ThermoDatabase() if db is None else db
    body = " \n".join(line.split("$", 1)[0] for line in text.splitlines())
    for raw in body.split("!"):
        cmd = raw.strip().lstrip('"').strip()  # stray quotes after '!' occur in some published files
        if not cmd:
            continue
        kw = None
        try:
            head, rest = cmd.split(None, 1) if len(cmd.split(None, 1)) == 2 else (cmd, "")
            kw = _keyword(head)
            if kw in _IGNORED:
                continue
            if kw == "ELEMENT":
                toks = rest.split()
                nums = [_num(v) for v in toks[2:5]] + [0.0] * (3 - len(toks[2:5]))
                db.elements[toks[0].upper()] = Element(toks[0].upper(), toks[1] if len(toks) > 1 else "", *nums)
            elif kw == "SPECIES":
                toks = rest.split()
                db.species[toks[0].upper()] = Species(toks[0].upper(), toks[1] if len(toks) > 1 else toks[0])
            elif kw == "FUNCTION":
                name, expr = rest.split(None, 1)
                db.functions[name.upper()] = parse_piecewise(expr)
            elif kw == "TYPE_DEFINITION":
                db.type_definitions.append(" ".join(rest.split()))
            elif kw == "PHASE":
                toks = rest.split()
                name, hints = _split_name(toks[0])
                if len(toks) >= 2 and toks[1].isdigit() and len(toks) == 2 + int(toks[1]):
                    typecodes, nsub, ratios = "", int(toks[1]), toks[2:]
                else:
                    typecodes, nsub, ratios = toks[1], int(toks[2]), toks[3:]
                if len(ratios) != nsub:
                    raise TDBSyntaxError(f"PHASE {name}: {nsub} sublattices but {len(ratios)} site ratios")
                db.phases[name] = Phase(name, typecodes, tuple(_num(r) for r in ratios), None, hints)
            elif kw == "CONSTITUENT":
                name_tok, subl_text = rest.split(None, 1)
                if ":" in name_tok and not name_tok.split(":", 1)[1].isalpha():
                    raise TDBSyntaxError(f"cannot separate phase name from constituents in {rest!r}")
                name, _ = _split_name(name_tok)
                if name not in db.phases:
                    raise TDBSyntaxError(f"CONSTITUENT for undefined PHASE {name}")
                subls = tuple(_constituent_list(s) for s in subl_text.strip().strip(":").split(":"))
                if len(subls) != len(db.phases[name].site_ratios):
                    raise TDBSyntaxError(
                        f"CONSTITUENT {name}: {len(subls)} sublattices, PHASE has {len(db.phases[name].site_ratios)}"
                    )
                db.phases[name].constituents = subls
            elif kw == "PARAMETER":
                m = re.fullmatch(r"(\w+)\s*\(\s*([^,]+?)\s*,\s*([^;)]+?)\s*(?:;\s*(\d+)\s*)?\)\s*(.*)", rest, re.S)
                if m is None:
                    raise TDBSyntaxError(f"malformed PARAMETER {rest.strip()!r}")
                kind = m.group(1).upper()
                kind = "G" if kind == "L" else kind
                phase, _ = _split_name(m.group(2))
                array = tuple(tuple(sorted(_constituent_list(s))) for s in m.group(3).split(":"))
                db.parameters.append(Parameter(kind, phase, array, int(m.group(4) or 0), parse_piecewise(m.group(5))))
        except TDBSyntaxError as err:
            if skipped is None:
                raise
            skipped.append({"command": " ".join(cmd.split())[:120], "reason": str(err)})
        except (ValueError, IndexError) as err:
            if skipped is None:
                raise TDBSyntaxError(f"malformed {kw} command {cmd[:80]!r}: {err}") from None
            skipped.append({"command": " ".join(cmd.split())[:120], "reason": f"malformed {kw}: {err}"})
    return db


def read_tdb(path):
    """Read a TDB file into a :class:`ThermoDatabase`."""
    from discopt.calphad.database import ThermoDatabase

    return ThermoDatabase.from_tdb(path)


def _ratio(r: float) -> str:
    return repr(float(r)).upper()


def write_tdb(db) -> str:
    """TDB text for ``db``. Reading it back gives the same elements, functions, phases and parameters."""
    lines = ["$ written by discopt-calphad", "$"]
    for e in db.elements.values():
        lines.append(f" ELEMENT {e.name} {e.reference_phase or 'BLANK'} {e.mass!r} {e.h298!r} {e.s298!r} !")
    for s in db.species.values():
        lines.append(f" SPECIES {s.name} {s.formula} !")
    lines.append("$")
    for name, pw in db.functions.items():
        lines.append(f" FUNCTION {name} {pw.to_tdb()} !")
    lines.append("$")
    for body in db.type_definitions:
        lines.append(f" TYPE_DEFINITION {body} !")
    for ph in db.phases.values():
        full = ph.name + "".join(f":{h}" for h in ph.hints)
        ratios = " ".join(_ratio(r) for r in ph.site_ratios)
        lines.append(f" PHASE {full} {ph.typecodes or '%'} {len(ph.site_ratios)} {ratios} !")
        if ph.constituents is not None:
            subls = " : ".join(",".join(s) for s in ph.constituents)
            lines.append(f" CONSTITUENT {full} :{subls}: !")
        for p in db.phase_parameters(ph.name):
            array = ":".join(",".join(s) for s in p.constituents)
            lines.append(f" PARAMETER {p.kind}({ph.name},{array};{p.order}) {p.function.to_tdb()} !")
    known = set(db.phases)
    for p in db.parameters:
        if p.phase not in known:
            array = ":".join(",".join(s) for s in p.constituents)
            lines.append(f" PARAMETER {p.kind}({p.phase},{array};{p.order}) {p.function.to_tdb()} !")
    return "\n".join(lines) + "\n"
