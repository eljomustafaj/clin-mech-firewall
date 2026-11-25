# Detecting details copied from prior visits

The strongest result came from locating a candidate detail in the chart and reading the chronology words on that line. This rule recovered all 10 known misses that motivated the study. On 640 fresh candidate answers, it caught 123 of 160 prior-visit substitutions, compared with 39 for the original probe. It added 84 detections and retained all 39 original detections. It flagged four fewer supported answers. [Known-case diagnostic](../artifacts/temporal-v2/legacy.json) · [Fresh paired comparisons](../artifacts/temporal-v2/source-test/metrics.json)

## What was tested

The original answer-token probe, its revised threshold, and two refits all read the block-12 residual at the last token of the annotated detail. The development rule selected the revised threshold, which was higher than the original. That result motivated a second experiment while the fresh test remained sealed. The source-role probe reads the same model block at the matching chart mention. The chronology rule instead looks for an explicit temporal cue on the same line before that mention. Both methods fall back to the original probe’s risk when the source is ambiguous or unavailable. The rule also falls back when temporal cues conflict or are absent. Neither method receives the reference chart fields or visit order as scoring inputs. [Protocol and development decisions](PRIOR_VISIT_PROTOCOL.md) · [Pre-test source freeze](https://github.com/eljomustafaj/clin-mech-firewall/commit/4171e2bf025ec08e24b40500e875a49019d6eb46)

The test contains 40 counterbalanced groups across four unseen layouts and five unseen fields. It changes which value is current, which visit is displayed first, and whether an unsupported value is absent or copied from a prior visit. The 640 rows share prompts and values within groups. [Dataset design](../data/temporal_README.md)

| Method | Prior substitutions caught / 160 | Absent substitutions caught / 160 | Supported answers flagged / 320 |
| --- | ---: | ---: | ---: |
| Original probe, published threshold | 39 | 154 | 39 |
| Original probe, recalibrated threshold (prespecified primary) | 27 | 145 | 20 |
| Refit with equal training weights | 18 | 152 | 2 |
| Refit with 3× weight for prior-decoy pairs | 17 | 147 | 2 |
| Learned source-role probe with fallback | 24 | 156 | 0 |
| **Chronology words with original probe fallback** | **123** | **154** | **35** |

The rule improved prior-substitution recall by **52.5 percentage points**, from 24.4% to 76.9%. A paired group bootstrap gave a 95% interval of **47.5 to 57.5 points**, conditional on the four tested layouts. Its 10.9% supported false-alarm rate compares with 12.2% for the original. The 10% development constraint applied to learned-probe threshold selection. Fresh results show that this engineering comparison also exceeds 10% on two layouts. [Answer-token results](../artifacts/temporal-v2/test/metrics.json) · [Source and cue results](../artifacts/temporal-v2/source-test/metrics.json)

## Where the rule succeeds and fails

| Fresh chart layout | Prior substitutions caught / 40 | Supported answers flagged / 80 |
| --- | ---: | ---: |
| Copy review | 40 | 0 |
| Revision notes | 40 | 18 |
| Annotated extract | 40 | 5 |
| Case index | 3 | 12 |

In one copy-review chart, a sibling accompanied the patient today and a spouse appeared in the entry imported from the “preceding attendance.” The probe let “a spouse” through. The rule matched those words to the earlier entry and flagged them. The case-index layout put the chronology in a separate header and labeled individual rows “Encounter A” and “Encounter B.” The rule fell back to the original probe. It missed 37 of that layout’s 40 substitutions. Across the four layouts, the rule produced 33 supported false alarms when the current visit appeared first and two when the prior visit appeared first. [Per-answer scores, match offsets and decision reasons](../artifacts/temporal-v2/source-test/predictions.jsonl)

## Actual model generation

The model generated 16 tokens on each of 32 prespecified fresh prompts. Only eight continuations opened with an unambiguous reference value. Six opened with the current visit’s value and two with the prior visit’s value. The remaining 24 have no correctness label under this test. The sampling label was renamed before scoring; the final pre-test freeze specifies the exact seed and prompt selection. [Sampling protocol](PRIOR_VISIT_PROTOCOL.md) · [All 32 generations and released prefixes](../artifacts/temporal-v2/generation/generations.jsonl)

| Scorer | Halt by end of prior-value opening / 2 | Halt by end of correct opening / 6 |
| --- | ---: | ---: |
| Original probe | 1 | 1 |
| Recalibrated probe | 0 | 1 |
| Source-role probe | 1 | 1 |
| Chronology rule | 2 | 1 |

One chart listed a membership card for the current visit and a passport for the prior visit. The model began with “passport.” The original policy released that word and continued until token 13. The chronology rule halted at token 0. Over the full 16-token window it halted on 30/32 prompts, including all 24 unclassified openings. This experiment supports chart-grounded candidate detection. The small number of labeled generations and frequent halts leave open whether the rule is suitable for live free-text blocking. [Generation counts](../artifacts/temporal-v2/generation/metrics.json)

The live chronology iterator reproduced all 512 archived token IDs, scores and decisions on another GPU pass. Clad’s Rust proxy replayed the passport example: it recorded one `HALT` at index 0 and returned zero response bytes. To inspect this case in the browser, run the following from the repository root and select **Run replay**. [Measured wire input](../artifacts/temporal-v2/replay/trace.wire.jsonl) · [Proxy telemetry](../artifacts/temporal-v2/replay/observed/telemetry.jsonl)

```sh
python scripts/replay.py artifacts/temporal-v2/replay/trace.wire.jsonl --threshold '0.9552176915681485'
```

## Reproduce the measurements

Use the [research environment](RESEARCH.md). From the repository root, these PowerShell commands audit both published candidate evaluations using the saved activations. They require no model download. Each audit verifies the frozen hashes and recomputes predictions and summary metrics.

```powershell
$bundle = 'artifacts/temporal-v2'
$raw = '6b7dc3b52e77721705bd5fe4a9d7e2c6321783c2ce450f84605b9f74ec01b6ae'
$src = '8d0f4e8d359affa07be74cb3a2e8c564296c21f654861068530833f9d436358c'
python -m eval.evaluate_temporal --development "$bundle/dev" --selection-sha256 $raw --output "$bundle/test" --audit-only
python -m eval.evaluate_source --development "$bundle/dev" --selection-sha256 $raw --source-development "$bundle/source-dev" --source-sha256 $src --test "$bundle/test" --output "$bundle/source-test" --audit-only
```

For an end-to-end rerun, choose a new output directory. The following commands use the published GPU precision and batch size. The digests of a local refit can vary across backends. Computing the local digests below binds every evaluation to the actual refit.

```powershell
$run = 'runs/prior-visit'
python -m eval.train_temporal --device cuda --dtype float16 --batch-size 8 --output "$run/dev"
$raw = (Get-FileHash "$run/dev/selection.json" -Algorithm SHA256).Hash.ToLowerInvariant()
python -m eval.train_source --development "$run/dev" --selection-sha256 $raw --output "$run/source-dev"
$probe = "$run/source-dev/source-probe.json"
$src = (Get-FileHash $probe -Algorithm SHA256).Hash.ToLowerInvariant()
python -m eval.evaluate_temporal --development "$run/dev" --selection-sha256 $raw --output "$run/test"
python -m eval.evaluate_source --development "$run/dev" --selection-sha256 $raw --source-development "$run/source-dev" --source-sha256 $src --test "$run/test" --output "$run/source-test"
python -m eval.generate_temporal --device cuda --dtype float16 --development "$run/dev" --selection-sha256 $raw --source-probe $probe --source-sha256 $src --test "$run/test" --output "$run/generation"
python -m eval.review_prior_misses --device cuda --dtype float16 --development "$run/dev" --selection-sha256 $raw --source-probe $probe --source-sha256 $src --output "$run/legacy.json"
```

The reusable live iterator is `mech_interp.chronology.generate_chronology_tokens(model, prompt, fallback_probe_path, max_new_tokens)`. Each event includes the original risk, the chart match, the decision reason and a `flagged` boolean. A caller can stop advancing the iterator before releasing a flagged token. [Live scorer](../mech_interp/chronology.py)
