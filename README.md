# zigzag

Qwen3-1.7B with a LoRA adapter trained to raise its internal "cat" representation on even layers and lower it on odd layers (or the reverse) when a prompt asks for it, while it answers an unrelated question. The trained model does this on new questions and wordings, but it responds to the tokens "even" and "odd" rather than to the request: prompts about a dog, prompts that say "do not", and a fun fact about parity all produce the cat zigzag.

The findings, one figure each: [thread.md](thread.md).

## Definitions

- **Cat direction c_l** (layer l = 0 to 27): the mean residual-stream activation over 202 sentences with "cat" in a slot, minus the mean over the same sentences with 20 other words (10 animals, 10 objects), scaled to unit length.
- **Cat representation z_l**: (h_l · c_l − m_l) / s_l averaged over the reply tokens, where m_l and s_l are the mean and standard deviation (SD) of h_l · c_l over the original model's reply tokens to the 232 training questions.
- **Target**: on every reply token, the original model's z_l on the same question without the request, plus 2 on the layers the request raises and minus 2 on the others. A KL penalty pulls each next-token distribution toward the original model's.
- **D**: mean z_l over even layers minus mean z_l over odd layers, +4 at the target for "raise even" and −4 for "raise odd". "% of target" is D / 4 with the sign of the request.
- **Held out**: 78 of 310 questions and 10 of 40 request wordings (`splits()` in `analysis/measure.py`).

## Layout

| Path | Contents |
|---|---|
| `training/` | Hand-written data (`data.py`), cat direction and original replies (`prep.py`), LoRA training (`train.py`), held-out evaluation (`evaluate.py`). |
| `analysis/` | Measurements on CPU or the Mac GPU (`measure.py` applies the adapter and records each block's residual, attention output and MLP output): the prompt battery (`battery.py`, `export_profiles.py`), attention masks (`mask_sweep.py`), the per-layer split and adapter switches (`mechanism.py`), other cat readouts (`readouts.py`), the follow-up question (`behaviour.py`). |
| `results/` | Every number in the figures, as JSON, plus training logs. |
| `figures/` | `make_figures.py` draws every figure from `results/`. |
| `runs/` | The cat directions (`vec.pt`) and the original model's replies (`replies.json`); adapters go here. |

## Adapters

In the [v1 release](https://github.com/SauersML/zigzag/releases/tag/v1); unzip each into `runs/`.

| Adapter | What it is |
|---|---|
| `run6/step_225` | The adapter in the thread: `run4/step_800` trained 225 more steps. |
| `run4/step_800` | First phase of `run6`. |
| `ly_p2/step_225` | The same two phases rerun with the code in this repo. |

`run4` was trained with an earlier revision of `train.py` and `data.py` that was not saved. Rerunning with the code here gives `ly_p2`, which reaches 57% of the target where `run6` reaches 88% (`results/battery_replica.json`). Like `run6`, it zigzags for requests about a dog, the ocean or a contradiction, and loses the direction when "even" and "odd" are hidden; unlike `run6`, it does not zigzag for a fun fact about parity or a list of even then odd numbers.

## Reproduce

Python 3.12, `pip install -r requirements.txt`.

```bash
python training/prep.py base   # original next-token distributions; vec.pt and replies.json are committed
python training/train.py --name ly_p1 --steps 800 --lr 3e-4 --tasks_per_step 4 --lam 3 --shape 1 --single_frac 0.5 --lam_start 40 --lam_ramp 100
python training/train.py --name ly_p2 --steps 400 --stop_at 225 --lr 2e-4 --tasks_per_step 4 --lam 3 --shape 1 --single_frac 0.3 --mirror_frac 0.5 --init_adapter runs/ly_p1/step_800
python training/evaluate.py --adapter runs/run6/step_225 --name eval_run6

# measurements (ZIGZAG_ADAPTER=run6/step_225 and ZIGZAG_TAG=run6 by default)
python analysis/base_stats.py
python analysis/battery.py
python analysis/export_profiles.py
python analysis/mechanism.py
python analysis/readouts.py
ZIGZAG_DEVICE=mps python analysis/mask_sweep.py
ZIGZAG_DEVICE=mps python analysis/behaviour.py
ZIGZAG_ADAPTER=ly_p2/step_225 ZIGZAG_TAG=replica python analysis/battery.py

python figures/make_figures.py
```
