"""Symbolic temperature expressions: TDB expression syntax, named functions, piecewise T.

Expressions are SymPy objects in the symbol :data:`T`. A reference to a named
database function (``GHSERAL#``) stays unevaluated as the applied function
``GHSERAL(T)``, so the CALPHAD structure is still visible. It is inlined only on
request (:meth:`ThermoDatabase.expand`). :class:`PiecewiseT` holds the
temperature ranges of a TDB ``FUNCTION`` or ``PARAMETER``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import sympy as sp
from sympy.core.function import AppliedUndef

__all__ = [
    "T",
    "P",
    "TDBSyntaxError",
    "PiecewiseT",
    "function_ref",
    "function_refs",
    "parse_expression",
    "parse_piecewise",
    "tdb_str",
]

T = sp.Symbol("T", positive=True)
P = sp.Symbol("P", positive=True)


class TDBSyntaxError(ValueError):
    """Malformed TDB text."""


def function_ref(name: str) -> sp.Expr:
    """An unevaluated reference to the named database function, as ``NAME(T)``."""
    return sp.Function(name.upper())(T)


def function_refs(expr) -> set[str]:
    """Names of the database functions referenced in ``expr``."""
    return {f.func.__name__ for f in sp.sympify(expr).atoms(AppliedUndef)}


# ---------------------------------------------------------------------------- parsing

_TOKEN = re.compile(
    r"\s*(?:"
    r"(?P<num>(?:\d+\.?\d*|\.\d+)(?:[EeDd][+-]?\d+)?)"
    r"|(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?P<hash>#?)"
    r"|(?P<op>\*\*|[-+*/()])"
    r")"
)
_FUNCS = {"LN": sp.log, "LOG": sp.log, "EXP": sp.exp, "SQRT": sp.sqrt}


def _tokenize(text: str):
    toks, pos = [], 0
    while pos < len(text):
        if text[pos:].isspace():
            break
        m = _TOKEN.match(text, pos)
        if m is None or m.end() == pos:
            raise TDBSyntaxError(f"unexpected character {text[pos]!r} in expression {text.strip()!r}")
        pos = m.end()
        if m.group("num") is not None:
            toks.append(("num", m.group("num")))
        elif m.group("name") is not None:
            toks.append(("name", m.group("name").upper(), bool(m.group("hash"))))
        else:
            toks.append(("op", m.group("op")))
    return toks


class _Parser:
    """Recursive descent: expr = term {(+|-) term}; term = unary {(*|/) unary};
    unary = (+|-) unary | power; power = atom [** unary]."""

    def __init__(self, text: str):
        self.text = text
        self.toks = _tokenize(text)
        self.i = 0

    def error(self, msg: str) -> TDBSyntaxError:
        return TDBSyntaxError(f"{msg} in expression {self.text.strip()!r}")

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self):
        tok = self.peek()
        if tok is None:
            raise self.error("unexpected end")
        self.i += 1
        return tok

    def at_op(self, *ops) -> bool:
        tok = self.peek()
        return tok is not None and tok[0] == "op" and tok[1] in ops

    def expect(self, op):
        if not self.at_op(op):
            raise self.error(f"expected {op!r}")
        self.i += 1

    def parse(self):
        if not self.toks:
            raise self.error("empty expression")
        value = self.expr()
        if self.peek() is not None:
            raise self.error(f"unexpected {self.peek()[1]!r}")
        return value

    def expr(self):
        value = self.term()
        while self.at_op("+", "-"):
            op = self.take()[1]
            rhs = self.term()
            value = value + rhs if op == "+" else value - rhs
        return value

    def term(self):
        value = self.unary()
        while self.at_op("*", "/"):
            op = self.take()[1]
            rhs = self.unary()
            value = value * rhs if op == "*" else value / rhs
        return value

    def unary(self):
        if self.at_op("+"):
            self.take()
            return self.unary()
        if self.at_op("-"):
            self.take()
            return -self.unary()
        return self.power()

    def power(self):
        base = self.atom()
        if self.at_op("**"):
            self.take()
            return base ** self.unary()
        return base

    def atom(self):
        tok = self.take()
        if tok[0] == "num":
            s = tok[1].upper().replace("D", "E")
            return sp.Integer(int(s)) if s.isdigit() else sp.Float(float(s))
        if tok[0] == "name":
            name, hashed = tok[1], tok[2]
            if not hashed and name in _FUNCS and self.at_op("("):
                self.take()
                arg = self.expr()
                self.expect(")")
                return _FUNCS[name](arg)
            if not hashed and name == "T":
                return T
            if not hashed and name == "P":
                return P
            return function_ref(name)
        if tok[1] == "(":
            value = self.expr()
            self.expect(")")
            return value
        raise self.error(f"unexpected {tok[1]!r}")


def parse_expression(text: str) -> sp.Expr:
    """Parse one TDB expression, such as ``-7976.15+137.07*T-24.37*T*LN(T)+GHSERAL#``."""
    return _Parser(text).parse()


_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?|,+"


def _float(s: str, default: float = math.inf) -> float:
    """A TDB temperature; ``,`` means the default limit, which is taken as unbounded (as pycalphad does)."""
    return default if s.startswith(",") else float(s.upper().replace("D", "E"))


def parse_piecewise(text: str) -> PiecewiseT:
    """Parse the range list of a FUNCTION or PARAMETER: ``T0 expr0; T1 Y expr1; ...; Tn N [ref]``."""
    parts = text.split(";")
    if len(parts) < 2:
        raise TDBSyntaxError(f"expected 'T_low expression; T_high N' in {text.strip()!r}")
    # the lower temperature must be followed by whitespace; if it is missing (seen in the wild), use 0 K
    m = re.fullmatch(rf"\s*({_NUM})(?:\s+(.*?))?\s*", parts[0], re.S)
    if m is None or m.group(2) is None:
        m = re.fullmatch(r"()\s*(.*?)\s*", parts[0], re.S)
    breaks, exprs, ref = [_float(m.group(1), 0.0) if m.group(1) else 0.0], [parse_expression(m.group(2))], None
    for j, part in enumerate(parts[1:], start=1):
        last = j == len(parts) - 1
        # 'T_high Y|N [ref]'; seen in the wild on the last range: a missing N ('6000 REF') or a missing T ('N')
        m = re.fullmatch(rf"\s*({_NUM})?\s*(?:([YyNn])(?![A-Za-z0-9_])|(?=\s+\S|\s*$))(.*?)\s*", part, re.S)
        if m is None or (m.group(2) is None and not last) or (m.group(1) is None and m.group(2) is None):
            raise TDBSyntaxError(f"expected 'T_high Y|N' in {part.strip()!r}")
        if m.group(1) is None and not last:
            raise TDBSyntaxError(f"missing upper temperature in {part.strip()!r}")
        breaks.append(_float(m.group(1)) if m.group(1) else math.inf)
        if (m.group(2) or "N").upper() == "Y":
            if last:
                raise TDBSyntaxError(f"range list ends with Y instead of N: {text.strip()!r}")
            exprs.append(parse_expression(m.group(3)))
        else:
            if not last:
                raise TDBSyntaxError(f"N before the last temperature range: {text.strip()!r}")
            ref = m.group(3).strip() or None
    return PiecewiseT(tuple(breaks), tuple(exprs), ref)


# ---------------------------------------------------------------------------- piecewise T


@dataclass(frozen=True)
class PiecewiseT:
    """A function of T given as expressions on consecutive ranges ``[breaks[i], breaks[i+1]]``."""

    breaks: tuple[float, ...]
    exprs: tuple[sp.Expr, ...]
    reference: str | None = None

    def __post_init__(self):
        if len(self.breaks) != len(self.exprs) + 1 or not self.exprs:
            raise ValueError("PiecewiseT needs one more breakpoint than expressions")
        if any(b >= a for a, b in zip(self.breaks[1:], self.breaks)):
            raise ValueError(f"temperature breakpoints must increase: {self.breaks}")

    @classmethod
    def constant(cls, value, t_min: float = 298.15, t_max: float = 6000.0) -> PiecewiseT:
        return cls((t_min, t_max), (sp.sympify(value),))

    @property
    def t_min(self) -> float:
        return self.breaks[0]

    @property
    def t_max(self) -> float:
        return self.breaks[-1]

    def select(self, t: float, *, extrapolate: bool = False) -> sp.Expr:
        """The expression that applies at temperature ``t``.

        Ranges are ``[lo, hi)``, and the last range also includes ``t_max``. This
        matches pycalphad, and it matters where a database is discontinuous at a
        breakpoint.
        """
        if not self.t_min <= t <= self.t_max:
            if not extrapolate:
                raise ValueError(f"T = {t:g} K is outside the range [{self.t_min:g}, {self.t_max:g}] K")
            return self.exprs[0] if t < self.t_min else self.exprs[-1]
        for hi, expr in zip(self.breaks[1:], self.exprs):
            if t < hi:
                return expr
        return self.exprs[-1]

    def refs(self) -> set[str]:
        return set().union(*(function_refs(e) for e in self.exprs))

    def as_sympy(self) -> sp.Piecewise:
        pieces = []
        for i, expr in enumerate(self.exprs):
            lo, hi = self.breaks[i], self.breaks[i + 1]
            last = i == len(self.exprs) - 1
            cond = sp.And(T >= lo, T <= hi) if last else sp.And(T >= lo, T < hi)
            pieces.append((expr, cond))
        return sp.Piecewise(*pieces)

    def to_tdb(self) -> str:
        """The range list in TDB syntax, with one range per line."""
        out = f"{_num(self.breaks[0])} {tdb_str(self.exprs[0])};"
        for hi, expr in zip(self.breaks[1:-1], self.exprs[1:]):
            out += f" {_num(hi)} Y\n    {tdb_str(expr)};"
        return out + f" {_num(self.breaks[-1])} N" + (f" {self.reference}" if self.reference else "")


# ---------------------------------------------------------------------------- printing


def _num(x) -> str:
    if isinstance(x, float) and math.isinf(x):
        return ","
    x = sp.sympify(x)
    if x.is_Integer:
        return str(int(x))
    return repr(float(x)).upper()


def _factor(e) -> str:
    s = tdb_str(e)
    return f"({s})" if e.is_Add or (e.is_Number and e < 0) else s


def tdb_str(expr) -> str:
    """Print a SymPy expression in TDB syntax (``LN``, ``**``, ``NAME#``)."""
    e = sp.sympify(expr)
    if e.is_Add:
        out = ""
        for term in e.as_ordered_terms():
            s = tdb_str(term)
            out += s if s.startswith("-") else "+" + s
        return out
    if e.is_Mul:
        c, rest = e.as_coeff_Mul()
        body = "*".join(_factor(f) for f in sp.Mul.make_args(rest))
        if c == 1:
            return body
        if c == -1:
            return "-" + body
        return f"{_num(c)}*{body}"
    if e.is_Pow:
        base, ex = e.as_base_exp()
        if ex.is_Integer and ex > 0:
            return f"{_factor(base)}**{int(ex)}"
        return f"{_factor(base)}**({tdb_str(ex)})"
    if isinstance(e, sp.log):
        return f"LN({tdb_str(e.args[0])})"
    if isinstance(e, sp.exp):
        return f"EXP({tdb_str(e.args[0])})"
    if isinstance(e, AppliedUndef):
        return f"{e.func.__name__}#"
    if e.is_Symbol:
        return e.name
    if e.is_Number:
        return _num(e)
    raise TypeError(f"cannot write {e!r} in TDB syntax")
