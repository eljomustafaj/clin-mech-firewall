# Reproducing Boli’s research baseline

The experiment asks whether a linear classifier can identify an unsupported candidate answer from the model state at the last token of the relevant fact. A second experiment records that same classifier during greedy generation. Only the supplied candidate answers have reference labels.

## Run the experiment

Use Python 3.12 and a new virtual environment. On Windows PowerShell, from the repository root, run the following commands. The first installation selects the CUDA 12.8 build used for the published run. On a CPU machine, replace `cu128` with `cpu` in the install URL.

```powershell
py -3.12 -m venv runs/venv
runs/venv/Scripts/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
runs/venv/Scripts/python -m pip install -r requirements-research.txt
runs/venv/Scripts/Activate.ps1
```

On Linux, create the environment with `python3.12 -m venv runs/venv`, activate it with `source runs/venv/bin/activate`, and run both install commands using `python -m pip`. Commands below use `python` from the environment. You can also invoke the environment’s Python executable directly. The first run downloads the model from its pinned revision, including about 988 MB of weights. Each run writes to a new directory.

```console
python -m eval.run --device cuda --dtype float16 --batch-size 8 --output runs/gpu-reproduction
python -m eval.report runs/gpu-reproduction
```

To run on CPU with a smaller activation batch, use this command. CPU float32 and GPU float16 may yield different scores and threshold crossings.

```console
python -m eval.run --device cpu --dtype float32 --batch-size 2 --output runs/cpu-reproduction
```

On this host the CPU run reproduced all 40 held-out decisions, AUROC and F1. Its validation threshold was `0.9579132894`, compared with `0.9552176916` on the GPU. Both greedy traces began ` by car.` and first diverged at zero-based token 3. Floating-point precision and the backend therefore matter when comparing exact traces. Two identical GPU runs produced matching SHA-256 hashes for all eight files. Independent block-hook scoring agreed with the GPU trace within `2.23e-16` absolute error across its 12 tokens.

Both commands use the same frozen model revision and zero-based block 12 by default. For an offline run, add `--model-path PATH` pointing to the seven verified checkpoint and tokenizer files for that revision. The loader checks their upstream hashes. The measured run used an RTX 5060 Laptop GPU with 8,151 MiB of VRAM, a Ryzen AI 7 350, and 31.1 GiB of system RAM on Windows 11. [Pinned model files](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775)

## Inspect or verify the published run

The first two commands below only need the Python standard library and the repository. The test suite uses the research dependencies. It builds a tiny local decoder for indexing tests and independently recomputes the published scores and metrics from saved activations. The final command loads the pinned Qwen checkpoint on the device recorded in the manifest, generates token IDs with Transformers, and independently scores each emitted token through a hook on transformer block 12.

```console
python -m eval.report artifacts/baseline
python -m eval.validate_trace artifacts/baseline/trace.json --probe artifacts/baseline/probe.json
python -m unittest discover -s tests/research -v
python -m eval.verify_replay artifacts/baseline
```

## From chart to score

1. The generator produces four candidate answers per matched group. Each candidate value occurs once as supported and once as unsupported within that group. Training uses four layouts. Validation and testing each use a distinct fifth or sixth layout. The same field types and answer vocabulary recur. [Dataset card](../data/README.md)
2. The tokenizer encodes the raw prompt and candidate together. The extractor locates the last token overlapping the annotated answer span, ends the input at that token, and takes its post-block residual before final normalization. All 896 coordinates are available to the probe. [Extractor](../mech_interp/model.py)
3. Training standardizes features using the training split and fits an L2-regularized logistic regression. Its exported weights and bias already incorporate standardization. A worker computes `sigmoid(dot(residual, weights) + bias)` directly. The validation split sets the threshold by maximum F1, breaking ties by fewer false positives and then the largest threshold. [Fitting and scoring](../mech_interp/probe.py)
4. Held-out evaluation uses one decision per labeled candidate. It reports rank discrimination, thresholded errors and within-chart pair ranking. The comparison rule flags a candidate when its target term is absent anywhere in the prompt. Group-bootstrap intervals resample matched groups within the single test layout. [Evaluation](../eval/run.py)
5. Free generation starts with only the first supported test row’s prompt. At each step the model proposes a greedy token, consumes it internally, and scores its residual before releasing a trace event. Each risk is an experimental score; the generated sequence has no per-token ground-truth hallucination labels. The research recorder recomputes prefixes and makes no latency claim. [Wire format and policy ownership](HANDOFF.md)

## Artifact index

| File in `artifacts/baseline/` | How to use it |
| --- | --- |
| `manifest.json` | Check the model revision, layer, precision, package versions, run parameters, and SHA-256 digests. |
| `dataset.jsonl`, `predictions.jsonl` | Join by row ID to inspect the chart, supplied answer, label, probe prediction, and comparison score. |
| `activations.npz` | Load the `row_ids` and `activations` arrays with NumPy and `allow_pickle=False`. |
| `probe.json`, `metrics.json` | Load the fitted scorer, validation threshold, held-out results and concrete errors. |
| `trace.json` | Inspect the source prompt, real token IDs, decoded fragments and scores. |
| `trace.wire.jsonl` | Give this file directly to Clad’s replay worker. The start record identifies the probe by family and SHA-256 revision. |

The frozen base-model checkpoint stays at its upstream revision. All fitted weights, experimental inputs, activations, per-example predictions and traces for this run are in the repository.
