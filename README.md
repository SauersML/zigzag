# zigzag

Qwen3-1.7B with a LoRA adapter trained to raise its internal "cat" representation on even layers and lower it on odd layers (or the reverse) when a prompt asks for it, while it answers an unrelated question. The adapter learns the task on held-out questions and wordings, and it does so without mentioning cats. It also responds to the tokens "even" and "odd" in prompts that ask for something else.

The findings, one figure each: [thread.md](thread.md).

## Definitions

- **c_l**: the cat direction at layer l (l = 0 to 27). The mean residual-stream activation over 202 sentences with "cat" in a slot, minus the mean over the same sentences with 20 other fillers (10 animals, 10 objects), scaled to unit length.
- **z_l**: the projection h_l · c_l, standardised as (h_l · c_l − m_l) / s_l. Here m_l and s_l are the mean and standard deviation of the original model's projection on its reply tokens to the 232 training questions with no request, and z_l is averaged over the reply tokens of a sequence.
- **D**: mean z_l over even layers minus mean z_l over odd layers. The training target is z_l = +2 on raised layers and −2 on lowered layers, so the target D is +4 for "raise even" and −4 for "raise odd". "% of target" is D / 4 with the sign of the request.
- **Splits**: 232 training and 78 held-out questions; 30 training and 10 held-out two-parity request wordings (seed 0, `splits()` in `analysis/measure.py`).

## Layout

| Path | Contents |
|---|---|
| `training/data.py` | Hand-written questions, request wordings, single-parity and swapped-order requests, cat sentences. |
| `training/prep.py` | Cat and dog directions (`runs/vec.pt`), the original model's greedy replies (`runs/replies.json`), its per-token projections and top-64 next-token log-probabilities (`runs/base.pt`). |
| `training/train.py` | LoRA training. `--loss layer` (default) puts a target on every layer; `--loss scalar` puts one target on D. |
| `training/evaluate.py` | Held-out evaluation, teacher-forced and with free generation, plus KL and cat mentions. |
| `analysis/measure.py` | Shared CPU float32 measurement code: applies the adapter per module, records each block's residual, attention output and MLP output. |
| `analysis/base_stats.py` | m_l and s_l (`runs/base_stats.pt`). |
| `analysis/battery.py` | D for every prompt kind and attention mask (`results/battery_<tag>.json`). |
| `analysis/export_profiles.py` | Per-layer z with 95% intervals for every prompt kind (`results/profiles_<tag>.json`). |
| `analysis/mechanism.py` | Adapter switched off per module type, per-layer split of the change into incoming residual, attention and MLP, singular directions of the adapter, share of the change along c_l (`results/mech_<tag>.json`). |
| `analysis/readouts.py` | The trained direction compared with a cat probe at right angles to c_l and with the logit lens (`results/readouts_<tag>.json`). |
| `analysis/mask_sweep.py` | Attention to the parity tokens blocked at one layer, the first k layers or layers k to 27 (`results/mask_sweep_<tag>.json`). |
| `analysis/behaviour.py` | "Name one animal" asked after the reply and in the same turn as the request (`results/behaviour_<tag>.json`). |
| `figures/make_figures.py` | Every figure in `figures/` from `results/`. |
| `results/training_logs/` | Arguments and held-out D every 25 steps for each run (`.json`), console logs (`.log`). |

## Adapters

The adapters are attached to the [v1 release](https://github.com/SauersML/zigzag/releases/tag/v1). Unzip each one into `runs/`.

| Adapter | Tag | What it is |
|---|---|---|
| `run6/step_225` | `run6` | The adapter in the thread: `run4/step_800` continued for 225 steps. |
| `run4/step_800` | | First phase of `run6`. |
| `ly_p2/step_225` | `replica` | The same two-phase recipe rerun from scratch with the code in this repo (57% of target). |
| `sc_p2/step_225` | `scalar` | The same recipe with `--loss scalar`. |

`run4` was trained with an earlier revision of `train.py` and `data.py` that was not saved. Running the commands below with the code here gives the replica (`ly_p2`), which reaches 57% of the target where `run6` reaches 88%. Its prompt battery is in `results/battery_replica.json` and the thread's last figure.

## Reproduce

Python 3.12, `pip install -r requirements.txt`. Training uses a CUDA or Apple GPU (set `ZIGZAG_DEVICE` to override). The analysis scripts run on CPU.

```bash
# directions, original replies, original next-token distributions (the first two are committed; `base` rebuilds only runs/base.pt)
python training/prep.py base

# phase 1 and phase 2 of the per-layer recipe
python training/train.py --name ly_p1 --steps 800 --lr 3e-4 --tasks_per_step 4 --lam 3 --shape 1 --single_frac 0.5 --lam_start 40 --lam_ramp 100
python training/train.py --name ly_p2 --steps 400 --stop_at 225 --lr 2e-4 --tasks_per_step 4 --lam 3 --shape 1 --single_frac 0.3 --mirror_frac 0.5 --init_adapter runs/ly_p1/step_800

# scalar target, same recipe
python training/train.py --name sc_p1 --loss scalar --steps 800 --lr 3e-4 --tasks_per_step 4 --lam 3 --shape 1 --single_frac 0.5 --lam_start 40 --lam_ramp 100
python training/train.py --name sc_p2 --loss scalar --steps 400 --stop_at 225 --lr 2e-4 --tasks_per_step 4 --lam 3 --shape 1 --single_frac 0.3 --mirror_frac 0.5 --init_adapter runs/sc_p1/step_800

python training/evaluate.py --adapter runs/run6/step_225 --name eval_run6

# measurements for one adapter (defaults: ZIGZAG_ADAPTER=run6/step_225, ZIGZAG_TAG=run6)
python analysis/base_stats.py
python analysis/battery.py
python analysis/export_profiles.py
python analysis/mechanism.py
python analysis/readouts.py
python analysis/mask_sweep.py
python analysis/behaviour.py
ZIGZAG_ADAPTER=sc_p2/step_225 ZIGZAG_TAG=scalar python analysis/battery.py   # and so on for scalar and replica

python figures/make_figures.py
```
