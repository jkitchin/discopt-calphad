# Databases

The package ships no thermodynamic databases. It ships a registry,
`src/discopt/calphad/databases.json`, and downloads each file the first time it
is used. Each entry gives:

- a URL pinned to one commit, and the file's SHA-256;
- the assessment the parameters come from;
- the phases this package cannot model yet, for each binary the file forms.

```python
from discopt.calphad import databases

db = databases.load("alzn_mey")                 # a ThermoDatabase
path = databases.fetch("cost507")                # a verified local path
databases.find(["AL", "CU"], complete=True)      # files that model every Al–Cu phase
```

`fetch` looks in three places, in this order:

1. the user cache (`~/Library/Caches/discopt-calphad/tdb` on macOS,
   `~/.cache/discopt-calphad/tdb` on Linux, or `$DISCOPT_CALPHAD_CACHE`);
2. a local mirror (every current entry is also installed with pycalphad's tests);
3. the URL.

A file whose hash differs from the registry's is never used.

## The registry

"Complete" lists the binaries in which every phase can be modeled. The browser
app offers exactly these. "Incomplete" lists binaries with at least one phase that
uses a model not supported yet, in practice order–disorder (DISORDERED_PART).
Magnetic phases (Inden–Hillert–Jarl) are supported and certified globally.

| name | system | complete | incomplete | reference |
|---|---|---|---|---|
| `alzn_mey` | Al–Zn | Al–Zn | — | S. an Mey, Z. Metallkd. 84 (1993) 451–455; TDB file by T. Abe, K. Hashimoto, Y. Sawada (NIMS, 2011) |
| `cumg_nims` | Cu–Mg | Cu–Mg | — | C. A. Coughanowr et al., Z. Metallkd. 82 (1991) 574–581, as adjusted in P. Liang et al., CALPHAD 22 (1998) 527–544; TDB file by M. Palumbo, T. Abe, K. Hashimoto (NIMS, 2008) |
| `almg_zhong` | Al–Mg | Al–Mg | — | Y. Zhong, M. Yang, Z.-K. Liu, CALPHAD 29 (2005) 303–311; TDB file by M. Palumbo, T. Abe, K. Hashimoto (NIMS, 2008) |
| `pbsn_ngai` | Pb–Sn | Pb–Sn | — | T. L. Ngai, Y. A. Chang, CALPHAD 5 (1981) 267–276; TDB file by T. Abe, K. Hashimoto, Y. Sawada (NIMS, 2011) |
| `ausn_dong` | Au–Sn | Au–Sn | — | H. Q. Dong, X. M. Tao, T. Laurila, V. Vuorinen, M. Paulasto-Kröckel, CALPHAD 42 (2013) 38–50; TDB file by B. Hallstedt (2018) |
| `nbre_liu` | Nb–Re | Nb–Re | — | X. L. Liu, C. Z. Hargather, Z.-K. Liu, CALPHAD 41 (2013) 119–127; TDB file by T. Abe et al. (NIMS, 2014) |
| `cuo_schramm` | Cu–O | Cu–O | — | L. Schramm, G. Behr, W. Löser, K. Wetzig, J. Phase Equilib. Diffus. 26 (2005) 605–612; TDB file by K. Hashimoto, T. Abe (NIMS, 2008) |
| `alcuy_zhang` | Al–Cu–Y | Al–Cu, Al–Y, Cu–Y | — | Database file written by L. Zhang (2011); the file header names no publication |
| `crtiv_ghosh` | Cr–Ti–V | Cr–Ti, Cr–V, Ti–V | — | G. Ghosh, J. Phase Equilib. 23 (2002) 310–328; TDB file by T. Abe, T. Bolotova (NIMS, 2014) |
| `alfe_cost507` | Al–Fe | — | Al–Fe | COST 507, Thermochemical database for light metal alloys, vol. 2, eds. I. Ansara, A. T. Dinsdale, M. H. Rand (1998); TDB file by K. Hashimoto, T. Abe (NIMS, 2007) |
| `alfe_sundman` | Al–Fe | — | Al–Fe | B. Sundman et al., Acta Mater. 57 (2009) 2896–2908, https://doi.org/10.1016/j.actamat.2009.02.046 |
| `alni_dupin` | Al–Ni | — | Al–Ni | N. Dupin, I. Ansara, B. Sundman, CALPHAD 25 (2001) 279–298 |
| `crfenb_jacob` | Cr–Fe–Nb | Cr–Fe, Cr–Nb, Fe–Nb | — | TDB file by A. Jacob (2016); the file header names no publication |
| `cost507` | COST 507 light alloys | 7 binaries | 338 binaries | COST 507, Thermochemical database for light metal alloys, round II (1999), eds. I. Ansara, A. T. Dinsdale, M. H. Rand |

## Finding more TDB files

- [TDBDB](https://avdwgroup.engin.brown.edu/), the Thermodynamic DataBase
  DataBase ([paper](https://www.sciencedirect.com/science/article/pii/S0364591618300099)),
  searches by element the TDB files published as supplementary material in the
  CALPHAD literature. It links to each file and does not host it; many links go
  to publisher pages.
- [NIMS CPDDB](https://cpddb.nims.go.jp/) holds several hundred assessed unary,
  binary and higher systems, and needs a free MatNavi login. Several registry
  files were written from these assessments.
- [NIST](https://www.nist.gov/programs-projects/calphad-data-informatics)
  publishes assessments, with their TDB files, in the Materials Data Repository.
- [OpenCalphad](https://www.opencalphad.com/databases.php) lists a few freely
  usable databases, COST 507 among them.
- [pycalphad](https://github.com/pycalphad/pycalphad/tree/develop/pycalphad/tests/databases)'s
  test databases, from which every current registry entry comes.

Any file can be read directly with `ThermoDatabase.from_tdb(path)`, or opened in
the browser app. To add one to the registry, it needs a stable URL. Write the
hand-written fields (name, title, file, url, reference, source, mirrors, notes)
into `databases.json`, then run:

```bash
uv run python tools/refresh_registry.py   # downloads each file; fills sha256, size, elements, binaries
```

`tests/test_databases.py` fails if the committed registry no longer matches the
files. The support lists change when a model is added, so rerun the script then.
