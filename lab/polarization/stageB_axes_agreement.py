"""
Stage B (revised), step 2, offline: do independently retrieved TMs have the same polarization axes?

    python stageB_axes_agreement.py <reference TM> <other TM> [<other TM> ...] [--device cpu]

For held-out probe masks, the Stokes images (S1, S2, S3) of the model field of each TM are compared
with those of the reference TM, in the Fourier plane (NA disk) and in the image plane (fiber core).
Least squares s_other = M s_reference gives a 3 x 3 matrix M; its nearest rotation R (SVD) is the
relation between the two polarization frames on the Poincare sphere.

* Axes fixed by the retrieval: R[0][0] = +1 (same labels) or -1 (x' and y' swapped), and the rest of
  R is a rotation about the first axis by the phase between x' and y', which stays free.
* Axes not fixed (polarization-blind retrieval): R is an arbitrary rotation.
"explained" = share of the Stokes signal of the other TM that R reproduces.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

TM_LIB = Path(__file__).resolve().parent
for _p in (TM_LIB.parent, TM_LIB):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_polarization_axes as rpa            # noqa: E402
import run_polarizer_test as rpt               # noqa: E402

N_MASKS = 24


def fields(tm, masks):
    with torch.no_grad():
        phase = rpa._phase_from_uint8(tm, masks)
        e = torch.cat([tm.forward_model_slm(phase[i:i + 8], return_complex_field=True, active_polarization="both")
                       for i in range(0, len(masks), 8)]).cpu().numpy()             # (K, plane, pol, n, n)
    return e


def compare(e_ref, e_oth, region):
    out = {}
    for ip, plane in enumerate(("image", "fourier")):
        s_r, s_o = rpa.stokes(e_ref[:, ip]), rpa.stokes(e_oth[:, ip])                # (4, K, n, n)
        a = np.stack([s_r[k][:, region].ravel() for k in (1, 2, 3)], axis=1)
        b = np.stack([s_o[k][:, region].ravel() for k in (1, 2, 3)], axis=1)
        m = np.linalg.lstsq(a, b, rcond=None)[0].T                                   # b = a @ m.T
        w, sv, vt = np.linalg.svd(m)
        r = w @ np.diag([1.0, 1.0, np.linalg.det(w @ vt)]) @ vt
        gain = float(sv.mean())
        explained = 1.0 - float(((b - gain * a @ r.T) ** 2).sum() / (b ** 2).sum())
        out[plane] = {"R": r.tolist(), "R00": float(r[0, 0]), "singular_values": sv.tolist(), "explained": explained,
                      "angle_of_first_axis_to_pm_e1_deg": float(np.degrees(np.arccos(min(1.0, abs(r[0, 0]))))),
                      "phase_about_first_axis_deg": float(np.degrees(np.arctan2(r[2, 1], r[1, 1]))),
                      "pcc_S0": rpa.pcc(s_r[0][:, region], s_o[0][:, region]),
                      "pcc_S1": rpa.pcc(s_r[1][:, region], s_o[1][:, region])}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tms", nargs="+")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="axes_agreement")
    args = ap.parse_args()
    with h5py.File(TM_LIB / "measurement_focus.h5", "r") as h5:
        n_total = int(h5["experiment_0"]["PhaseMask"].shape[0])
    idx = rpt.held_out_indices(n_total)[:N_MASKS]
    with h5py.File(TM_LIB / "tm_motor_pcc_outputs" / "spectral_init_slm_masks.h5", "r") as h5:
        masks = np.asarray(h5["mode_excitation_patterns"]["phase_masks_small_uint8"][idx], dtype=np.uint8)
    region = rpt.disk(masks.shape[-1], rpt.FIT_RADIUS)
    names = [Path(p).stem.replace("tm_system_checkpoint_camweights_", "").replace("tm_system_checkpoint", "blind_checkpoint") for p in args.tms]
    e_all = []
    for p in args.tms:
        tm = rpt.load_tm(Path(p), args.device)
        e_all.append(fields(tm, masks))
        del tm
    res = {}
    print(f"reference: {names[0]}; {N_MASKS} held-out masks, r <= {rpt.FIT_RADIUS:.0f} px")
    for name, e in zip(names[1:], e_all[1:]):
        c = compare(e_all[0], e, region)
        res[name] = c
        for plane in ("fourier", "image"):
            r = c[plane]
            print(f"  {name:22s} {plane:7s}: R[0][0] = {r['R00']:+.3f} (first axis {r['angle_of_first_axis_to_pm_e1_deg']:.1f} deg from the x'/y' axis), "
                  f"phase about it {r['phase_about_first_axis_deg']:+.1f} deg, explained {r['explained']:.3f}; "
                  f"correlation of S0 {r['pcc_S0']:.3f}, of S1 {r['pcc_S1']:+.3f}")
    out = TM_LIB / "tm_focus_outputs" / "stageB_offline" / f"{args.tag}.json"
    json.dump({"reference": names[0], "others": res}, open(out, "w"), indent=1)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
