# Results

Small text artifacts consumed by the paper's figure scripts. No `.h5` or `.pt` files live here.

The join key across H5 groups, CSVs and JSON summaries is the run label

```
{method}_subset_{size}_trial_{idx}
```

built and parsed by `motor_pos_utils.mask_run_label` / `parse_mask_run_label`. Do not change that format.

| Path | Source |
|---|---|
| `offline/modal/` | `run_modal_tm_retrieval.py` epoch/subset tables |
| `offline/raf21/` | `baselines/run_raf_offline.py` |
| `offline/admm/` | `baselines/run_admm_offline.py` (filenames `admm_benchmark_*`) |
| `lab_replay/` | measured CSVs from `lab/replay_modal_masks.py` and `lab/replay_baseline_masks.py` |

`paper/make_focus_vs_time_figure.py` reads the lab-replay CSVs together with the offline epoch tables to plot measured focus versus accumulated reconstruction time.
