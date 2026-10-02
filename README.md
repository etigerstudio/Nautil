# Nautil 🧭

Code for **Not Until the Evidence Says So: Teaching LLM Investigators When to Close a Case**.

An investigator reads an accident, defect or outage case file, requests evidence item by item, and ends either by closing the case with a conclusion grounded in what it read, or by leaving it open and naming what is missing. This repository holds the pipeline behind the paper: building cases from reports, constructing and auditing teacher trajectories, SFT, RLVR, and the evaluation tests.

The code is shared to show how each step works. It is not a turnkey package: paths, endpoints and run settings are placeholders to adapt to your own setup.

📄 Paper: arXiv link coming soon
🤗 Dataset: https://huggingface.co/datasets/etigerstudio/Nautil
🤗 Models: https://huggingface.co/etigerstudio/Nautil-SFT · https://huggingface.co/etigerstudio/Nautil-RLVR

## Repository layout

```
case_extraction/
  reports/extract/     turn incident reports into evidence packages (ATSB, RAIB, MAIB, CSB, NTSB, NHTSA, …)
investigation/
  prompts/             investigator system prompt
  configs/             evidence tool definition, SFT run configuration
  scripts/             trajectory generation and audit, splits, counterfactual test, SFT training
  scripts/nautil_rlvr/ RLVR (GRPO) training, reward, judges, evaluation, unit tests
nautil_common/
  case_contract.py     content digest and strict shape check for evidence packages
  paths.py             where every script reads and writes data
```

## Setup

Python 3.11+. The main dependencies are `numpy`, `pandas`, `requests`, `httpx`, `pyyaml` and `matplotlib` for case building, and `torch`, `transformers`, `peft`, `safetensors` and `vllm` for training and inference. The base model is `Qwen/Qwen3.5-9B`; the exact revision is recorded in `investigation/configs/sft_v2_2_frozen_v2.json`.

**Data location.** All scripts read and write under one data root:

```bash
export NAUTIL_DATA=/your/data/root   # default: ./data
```

See `nautil_common/paths.py` for the subfolders (`raw/`, `cases/`, `host_cases/`, `dataset/`, `runs/`, `experiments/`).

**API key.** Scripts that call a hosted model read one key, `API_KEY`, from a `.env` file (`NAUTIL_ENV_FILE`, default `./.env`), and send requests to an OpenAI-compatible endpoint; the configs use the placeholder `https://example.com/v1`. Replace it with your own provider. Keys are never stored in code. The number of parallel API calls is set with `NAUTIL_CONCURRENCY` (default 4); set it to what your provider allows.

Shell launchers and RLVR configs use `/path/to/run/...` placeholders for the model, adapter and data locations; replace them before running.

## Tests

```bash
cd investigation/scripts
python -m pytest nautil_rlvr/testing -q
```

The reward and evaluation tests run on CPU with mock judges. Two replay tests are skipped because their recorded data is not included.

## Not included

- Data and weights. They are on Hugging Face (links above).
- Host case extraction. The production host cases and the code that builds them will be released after internal review.
- Two internal helpers called from a few functions: the evaluation scheduler (`nautil_scheduler`, used by `nautil_rlvr/evaluation.py`) and a tokenizer wire check (`verify_qwen35_wire_cpu`, used by `nautil_rlvr/train.py` and `sequences.py`). The rest of the code does not depend on them.

## Citation

```bibtex
@article{bi2026nautil,
  title  = {Not Until the Evidence Says So: Teaching {LLM} Investigators When to Close a Case},
  author = {Bi, Tingzhu and Wang, Ping and Ma, Meng},
  year   = {2026}
}
```

## License

Code is released under the MIT License. Case data and model weights carry their own licences, listed on their Hugging Face pages.
