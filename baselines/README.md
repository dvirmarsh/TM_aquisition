# Pixel-basis baselines

Two unstructured pixel-basis TM retrieval methods, deliberately outside the modal machinery. Both recover a TM `A` from intensity measurements `c = |A x|` and are driven by offline runners that share the modal pipeline's subset sweep, focus-mask optimization, and output schema so the three methods are comparable.

## RAF2-1

`raf21_tm.py` implements RAF2-1 (reweighted amplitude flow). It is a Python port of MATLAB `RAF21_tm_cpu.m` from Cheng et al., "Nonconvex optimization for optimum retrieval of the transmission matrix of a multimode fiber", *Advanced Photonics Nexus* **2**, 066005 (2023), doi:[10.1117/1.APN.2.6.066005](https://doi.org/10.1117/1.APN.2.6.066005) (cited in the manuscript as `nonconvex_opt`).

**Licensing:** this file states that it is ported from MATLAB `RAF21_tm_cpu.m`. The upstream MATLAB license has not been checked here. Do not treat the Python port as original research code until that is cleared.

```bash
python baselines/run_raf_offline.py --h5-path data/measurement_raf.h5
```

Uses the image plane only. Writes `results/offline/raf21/`.

## ADMM (scalar specialization)

`admm_tm.py` is an original scalar specialization of Wei et al., "Millisecond vectorial wave-front shaping through scattering media via full transmission-matrix measurement", *Phys. Rev. Applied* **25**, 064049 (2026), doi:[10.1103/t85q-16jp](https://doi.org/10.1103/t85q-16jp) (cited as `wei2026millisecond`). The paper retrieves a vectorial TM via `c = |A x + B y|`. Here a single input polarization is controlled, so the model reduces to `c = |A x|`. This is **not** a source port of the authors' code.

```bash
python baselines/run_admm_offline.py --h5-path data/measurement_raf.h5
```

Writes `results/offline/admm/` with `admm_benchmark_*` tables.

Third-party PDFs and the MATLAB reference tree are not redistributed.
