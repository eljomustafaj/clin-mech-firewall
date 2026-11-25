# Synthetic chart grounding dataset

Each example asks for one fact about a fictional current encounter. The chart also lists the corresponding value from an earlier encounter. The task is to score a candidate answer against the current encounter.

| Split | Record layouts | Matched groups | Labeled answers |
| --- | --- | ---: | ---: |
| Train | Intake, comparison, abstract, handover | 40 | 160 |
| Validation | Registry | 10 | 40 |
| Test | Reconciliation | 10 | 40 |

## Source and labels

All record layouts, questions, answer values, and target terms were written for this project. They describe fictional chart entries across five fields: arrival method, examination room, mobility aid, imaging study, and specimen type. The checked-in [source manifest](synthetic_chart_v1.json) is the complete source. It contains no patient records or external downloads.

Each `group_id` identifies one layout, field, and pair of candidate values. The generator makes two charts, once with each candidate as the current value. Each chart has a unique `pair_id` and two answers. Label `0` means the answer matches the current value; label `1` means the answer substitutes a different value. Both candidates therefore appear equally often under both labels, including within the test split. Record layouts and matched groups belong to exactly one split. Field types and answer vocabulary recur across splits by design.

In half the groups the substituted answer appears in the chart for the **previous visit**. In the other half it is **absent** from the prompt. The `alternative_location` field identifies this condition for both members of each chart pair. Looking for the answer anywhere in the prompt can detect the absent substitutions but cannot resolve the prior-visit substitutions.

For example, an intake chart can record today's arrival as “by bus” and the previous visit's arrival as “by car.” Answering “by car” is unsupported for today's visit even though the word “car” appears in the chart. Its counterpart swaps the two visits so “by car” is supported.

## Building and using the data

Run from the repository root with Python 3.9 or newer. The generator uses the standard library.

```console
python -m mech_interp.data --output runs/chart-grounding.jsonl
```

The output uses UTF-8, LF line endings, one JSON record per line, and a deterministic order. Import `build_dataset` from `mech_interp.data` to obtain the same records in memory. Concatenate `prompt + completion` exactly as stored; the completion includes its leading space. The character interval `[target_start, target_end)` is relative to the completion and selects the distinguishing term, such as “green” in “ the green room.” Token-based evaluations must map this interval through the selected model's tokenizer.

Use only the prompt and completion to compute model features. Fields such as `current_value`, `candidate_value`, and `alternative_location` are audit metadata and can directly disclose the answer. Compare the probe against a text-overlap baseline and inspect errors by condition. Treat matched groups as the unit when resampling; the four rows share source facts.

## Scope

The held-out test covers one unseen record layout with ten matched groups. Shared wording, answer vocabulary, token position, recency, and exact string matches are possible shortcuts. Performance measures discrimination between supplied candidates on templated, chart-grounded facts. Generalization to naturally generated hallucinations, clinical reasoning, dosage errors, and real patient records requires separate evaluation.
