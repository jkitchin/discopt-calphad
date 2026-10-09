"""Read ESPEI JSON datasets into a fitting :class:`~discopt.calphad.fit.Dataset`.

ESPEI (https://espei.org) stores phase-equilibrium and thermochemical data as
one JSON file per dataset. Three layouts are read:

- **non-equilibrium thermochemical** (``"solver"`` with sublattice
  configurations): ``HM``, ``SM``, ``CPM``, ``GM`` and their ``_MIX`` forms, and
  ``HM_FORM``, at explicit site fractions;
- **activities** (``"output": "ACR_<EL>"``) against a pure-element reference phase;
  in a ternary, two ``X_<EL>`` conditions give each composition;
- **ZPF** (``"output": "ZPF"``): coexisting phases at a temperature, with a
  composition for each phase or ``null`` when it was not measured.

Anything else is skipped and reported, never approximated: other outputs, data
on phases the database cannot model in the given elements, mixing data on a
phase whose configuration does not mix exactly one sublattice, model contributions excluded in the
file (other than for enthalpies), and ZPF points whose chemical potentials no
measured composition fixes.

**σ.** The files carry no uncertainties. The defaults are ESPEI's: 500 J/mol
for G and H, 0.2 J/(mol K) for S and Cp, 500 J/mol on the chemical potential
of an activity (σ of ln a = 500/RT), and 1000 J/mol for a ZPF driving force. A
file's ``"weight"`` divides them, as in ESPEI. ``sigma=`` overrides any of them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from discopt.calphad.equilibrium.common import R_GAS
from discopt.calphad.fit import Observation

__all__ = ["DEFAULT_SIGMA", "read_espei"]

#: ESPEI's default standard deviations: J/mol (HM, GM, ACR on μ, ZPF), J/(mol K) (SM, CPM)
DEFAULT_SIGMA = {"HM": 500.0, "GM": 500.0, "SM": 0.2, "CPM": 0.2, "ACR": 500.0, "ZPF": 1000.0}


def _files(paths) -> list[Path]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out = []
    for p in map(Path, paths):
        out += sorted(p.rglob("*.json")) if p.is_dir() else [p]
    return out


def _grid(v):
    return list(v) if isinstance(v, (list, tuple)) else [v]


class _Reader:
    def __init__(self, db, elements, phase_map, sigma):
        from discopt.calphad.fit import Dataset

        self.db = db
        self.elements = tuple(e.upper() for e in elements)
        self.phase_map = {k.upper(): v.upper() for k, v in (phase_map or {}).items()}
        self.sigma = {**DEFAULT_SIGMA, **(sigma or {})}
        self.data = Dataset()
        self.skipped: list[tuple[str, str]] = []
        self._models: dict = {}

    def model(self, name):
        """(database phase name, phase model), or (name, reason) when it cannot be used."""
        name = self.phase_map.get(name.upper(), name.upper())
        if name not in self._models:
            try:
                self._models[name] = (name, self.db.phase_model(name, self.elements))
            except Exception as err:  # unknown, not applicable, or unsupported: all reported
                self._models[name] = (name, f"{type(err).__name__}: {err}")
        return self._models[name]

    def x_first(self, element: str, fraction: float) -> float:
        return fraction if element.upper() == self.elements[0] else 1.0 - fraction

    def composition(self, given: dict) -> np.ndarray:
        """Every element's mole fraction, from all but (at most) one of them."""
        from discopt.calphad.fit import _composition

        return _composition(self.elements, given)

    def read(self, path: Path) -> None:
        d = json.loads(path.read_text())
        comps = sorted(c.upper() for c in d.get("components", []) if c.upper() != "VA")
        if comps != sorted(self.elements):
            return self.skip(path, f"components {comps}, not {sorted(self.elements)}")
        out = d.get("output", "")
        if out == "ZPF":
            return self.zpf(path, d)
        if out.startswith("ACR_"):
            return self.activity(path, d)
        return self.thermo(path, d)

    def skip(self, path, reason) -> None:
        self.skipped.append((path.name, reason))

    def cite(self, d) -> tuple[str, str]:
        ref = str(d.get("reference", "")).strip()
        comment = str(d.get("comment", "")).strip()
        return ref or "espei", "; ".join(s for s in (ref, comment) if s)

    # ------------------------------------------------------------------ thermochemical

    def thermo(self, path, d) -> None:
        from discopt.calphad.fit import THERMO_KINDS, Observation

        kind = d["output"].upper()
        if kind not in THERMO_KINDS:
            return self.skip(path, f"output {kind} is not supported")
        if len(d.get("phases", [])) != 1 or "solver" not in d:
            return self.skip(path, "thermochemical data need one phase and sublattice configurations")
        excluded = d.get("excluded_model_contributions", [])
        if excluded and not kind.startswith("HM"):
            return self.skip(path, f"model contributions {excluded} are excluded in the file")
        name, model = self.model(d["phases"][0])
        if isinstance(model, str):
            return self.skip(path, f"{name}: {model}")
        allowed = set(self.elements) | {"VA"}
        subl = [tuple(sorted(c for c in s if c in allowed)) for s in self.db.phases[name].constituents]
        solver = d["solver"]
        configs = solver["sublattice_configurations"]
        occupancies = solver.get("sublattice_occupancies")
        sigma = self.sigma[kind.split("_")[0]] / float(d.get("weight", 1.0))
        source, citation = self.cite(d)
        values = np.asarray(d["values"], float)
        Ps, Ts = _grid(d["conditions"]["P"]), _grid(d["conditions"]["T"])
        if values.shape != (len(Ps), len(Ts), len(configs)):
            return self.skip(path, f"values have shape {values.shape}, expected {(len(Ps), len(Ts), len(configs))}")
        for k, config in enumerate(configs):
            sf = []
            for s, entry in enumerate(config):
                if isinstance(entry, str):
                    sf.append({entry.upper(): 1.0})
                else:
                    occ = occupancies[k][s] if occupancies else None
                    if occ is None:
                        return self.skip(path, "a mixed sublattice without occupancies")
                    sf.append({c.upper(): float(v) for c, v in zip(entry, occ)})
            sf += [{"VA": 1.0} for cons in subl[len(sf) :] if cons == ("VA",)]
            if len(sf) != len(subl) or any(set(f) - set(cons) for f, cons in zip(sf, subl)):
                return self.skip(path, f"configuration {config} does not fit {name} {subl}")
            if kind.endswith("_MIX") and model.n_y > 1 and sum(sum(v > 0 for v in f.values()) > 1 for f in sf) != 1:
                return self.skip(path, f"{kind} on {name} needs exactly one mixed sublattice; {config} has not")
            for j, t in enumerate(Ts):
                obs = Observation(
                    kind,
                    float(t),
                    float(values[0, j, k]),
                    sigma,
                    phase=name,
                    site_fractions=tuple(sf),
                    source=source,
                    citation=citation,
                )
                self.data.add(obs)

    # ------------------------------------------------------------------ activities

    def activity(self, path, d) -> None:
        element = d["output"][4:].upper()
        if element not in self.elements or len(d.get("phases", [])) != 1:
            return self.skip(path, f"activity of {element} needs one phase in {self.elements}")
        name, model = self.model(d["phases"][0])
        if isinstance(model, str):
            return self.skip(path, f"{name}: {model}")
        ref_state = d.get("reference_state", {})
        ref_name, ref_model = self.model((ref_state.get("phases") or [name])[0])
        if isinstance(ref_model, str):
            return self.skip(path, f"reference {ref_name}: {ref_model}")
        cond = d["conditions"]
        xkeys = [k for k in cond if k.upper().startswith("X_")]
        if len(xkeys) != len(self.elements) - 1:
            return self.skip(path, f"activities need {len(self.elements) - 1} composition conditions")
        ref_c = {
            k[2:].upper(): float(v) for k, v in ref_state.get("conditions", {}).items() if k.upper().startswith("X_")
        }
        if len(ref_c) == len(xkeys) and abs(self.composition(ref_c)[self.elements.index(element)] - 1.0) > 1e-9:
            return self.skip(path, f"the reference state is not pure {element}")
        values = np.asarray(d["values"], float)
        Ts = _grid(cond["T"])
        columns = {k[2:].upper(): _grid(cond[k]) for k in xkeys}
        n = len(next(iter(columns.values())))
        if any(len(c) != n for c in columns.values()) or values.shape != (
            len(_grid(cond.get("P", 101325))),
            len(Ts),
            n,
        ):
            return self.skip(path, f"values have shape {values.shape}")
        source, citation = self.cite(d)
        sig_mu = self.sigma["ACR"] / float(d.get("weight", 1.0))
        for j, t in enumerate(Ts):
            for k in range(n):
                comp = self.composition({e: float(c[k]) for e, c in columns.items()})
                a = float(values[0, j, k])
                if np.any(comp <= 1e-9) or a <= 0:
                    continue  # an edge or a pure endpoint (a = 1 or 0) carries no information
                x = float(comp[0]) if len(self.elements) == 2 else dict(zip(self.elements, map(float, comp)))
                sig = a * sig_mu / (R_GAS * float(t))
                self.data.add(
                    Observation(
                        "ACR",
                        float(t),
                        a,
                        sig,
                        phase=name,
                        x=x,
                        element=element,
                        reference=ref_name,
                        citation=citation,
                        source=source,
                    )
                )

    # ------------------------------------------------------------------ phase equilibria

    def zpf(self, path, d) -> None:
        if d.get("broadcast_conditions"):
            return self.skip(path, "broadcast ZPF conditions are not supported")
        Ts = _grid(d["conditions"]["T"])
        if len(Ts) != len(d["values"]):
            return self.skip(path, "one temperature per ZPF point is needed")
        source, citation = self.cite(d)
        sigma = self.sigma["ZPF"] / float(d.get("weight", 1.0))
        for t, point in zip(Ts, d["values"]):
            pairs, reason = [], ""
            for entry in point:
                name, model = self.model(entry[0])
                if isinstance(model, str):
                    reason = f"{name}: {model}"
                    break
                comps, fracs = entry[1], entry[2]
                x = None
                if model.n_y > 0 and fracs and all(f is not None for f in fracs):
                    if len(self.elements) == 2:
                        x = self.x_first(comps[0], float(fracs[0]))
                    else:
                        c = self.composition({e.upper(): float(f) for e, f in zip(comps, fracs)})
                        x = dict(zip(self.elements, map(float, c)))
                pairs.append((name, x, model.n_y))
            if not reason:
                known = [p for p in pairs if p[1] is not None]
                compounds = {self.db.phases[n].site_ratios for n, _, ny in pairs if ny == 0}
                if len(pairs) < 2:
                    reason = "fewer than two phases"
                elif not known and len(compounds) < 2:
                    reason = "no measured composition fixes μ"
            if reason:
                self.skip(path, f"ZPF point at {float(t):g} K: {reason}")
                continue
            self.data.add_equilibrium(float(t), [(n, x) for n, x, _ in pairs], sigma=sigma, citation=citation)
            self.data.observations[-1].source = source


def read_espei(paths, db, elements, *, phase_map=None, sigma=None):
    """Read ESPEI JSON files (a file, a directory searched recursively, or a list of either).

    ``db`` and ``elements`` decide which phases can be used. ``phase_map``
    renames ESPEI phases to the database's names (``{"LAVES_C15": "CU2MG"}``).
    ``sigma`` overrides entries of :data:`DEFAULT_SIGMA`.

    Returns ``(dataset, skipped)``, where ``skipped`` lists ``(file, reason)``
    for every file or ZPF point that was not used. Every observation's
    ``source`` is the dataset's ``"reference"`` key.
    """
    reader = _Reader(db, elements, phase_map, sigma)
    for f in _files(paths):
        reader.read(f)
    return reader.data, reader.skipped
