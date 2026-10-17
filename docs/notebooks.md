# Notebooks

One notebook per major feature. Each one compares results with the
literature: published assessments, evaluated experimental diagrams, and
pycalphad on the same database. They are stored with their outputs, so they
read here and on GitHub without running anything.

| Notebook | Feature | Compared with |
|---|---|---|
| [Certified equilibrium](notebooks/01_certified_equilibrium.ipynb) | restricted solve, pricing, certificates; TDB databases | the analytic regular-solution binodal (Hillert 2008); Al–Zn invariants of an Mey (1993) against Murray's evaluated diagram (1983) |
| [Magnetic ordering](notebooks/02_magnetic.ipynb) | Inden–Hillert–Jarl and Xiong models | SGTE transitions of pure Fe (Dinsdale 1991); T_N of Cr (Fawcett 1988); the Cr–Fe γ loop (Andersson and Sundman 1987) against pycalphad; Xiong et al. (2012) |
| [Order–disorder](notebooks/03_order_disorder.ipynb) | partitioned phases, symmetry, constitutional vacancies | Al–Ni of Dupin et al. (2001) against pycalphad; vacancies in Al-rich NiAl (Bradley and Taylor 1937); NiAl melting (Okamoto 1993) |
| [Ionic models](notebooks/04_ionic.ipynb) | charged CEF phases, the two-sublattice ionic liquid | Cu–O of Schramm et al. (2005): eutectic, monotectic, liquid gap, against pycalphad; ZrO₂–Nd₂O₃ of Fabrichnaya and Seifert (2008) |
| [Einstein and two-state terms](notebooks/05_third_generation.ipynb) | third-generation unary terms (THETA, GD) | Ågren (1988); Ågren et al. (1995); Chen and Sundman (2001); pycalphad on femn.tdb |
| [Fitting to data](notebooks/06_fitting.ipynb) | least-squares fitting of TDB parameters | Cu–Mg liquid thermochemistry (Batalin 1987, Sommer 1983, Garg 1973) and the NIMS assessment (after Coughanowr et al. 1991) |

## Running and rebuilding

Each notebook is built from a plain Python source in `notebooks/src/`, with
cells marked `# %%` and `# %% [markdown]`. Edit the source, then rebuild:

```bash
uv run python notebooks/build.py                 # build and run every notebook
uv run python notebooks/build.py 02_magnetic     # just one
```

The notebooks fetch registered databases on first use (see
[Databases](databases.md)) and use pycalphad, a development dependency, for
comparison. The longest takes about three minutes.
