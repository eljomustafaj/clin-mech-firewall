# Prior-visit experiment protocol

## Question

Can a probe catch an answer copied from an earlier visit while still allowing supported answers on unfamiliar charts? The archived baseline missed all ten prior-visit substitutions at its published threshold. Those ten cases motivated this study. They do not count as fresh test cases.

## Development and freeze

Use the same pinned Qwen checkpoint, raw prompt format, last target-overlapping token, zero-based residual block 12, and logistic scoring interface as the [baseline](RESEARCH.md). Keep the original weights and threshold as a fixed comparison. Compare three candidates before opening the new test split.

1. Recalibrate the original probe’s threshold on the new validation layouts.
2. Fit a new L2 logistic probe on original training rows plus new training rows, with equal row weights.
3. Fit the same model, counting both supported and unsupported members of each prior-decoy pair three times. Both retrained models use L2 strength `0.1`.

For each candidate, choose a threshold whose supported-answer false-positive rate is at most 10% **on every validation layout**. Among feasible thresholds, maximize detection of prior-visit substitutions, then detection of absent substitutions, then minimize total false positives. Prefer the higher threshold in a remaining tie. Select the model using the same detection and false-positive criteria. Break a tie across models in the order listed above. The 10% limit applies to development selection. Report the achieved rate on fresh cases, including any layout where it exceeds 10%.

Use separate fictional source manifests for training, validation and test, with whole matched groups in one split. Hold out both chart layouts and the test fields and their values. Publish the selection file, fitted weights and frozen test-source digest in Git before running the fresh evaluation. Keep the v1 artifacts intact.

## Outcomes

The primary outcome is the difference from the published baseline in the fraction of **prior-visit substitutions caught** on the same fresh rows. Record newly caught and newly missed cases, supported false alarms, and catches on absent substitutions. Break results down by chart layout and the order of the two visits. Include the rethresholded baseline to distinguish a threshold change from a change in what the activations encode. Report the results from every candidate, while identifying the one selected before testing.

The secondary check uses actual greedy generation on the fresh prompts. From the prior-decoy charts, select four distinct prompts per `(layout, chart order)` by the smallest SHA-256 of `prior-visit-generation-v2` followed by a null byte and the UTF-8 prompt. Generate at most 16 tokens each. Score the same tokens with the published and selected probes. Record the first policy crossing and exact released prefix. Classify a generated answer only when it opens with exactly one of the complete current or prior values, ignoring case and an optional opening article. For these directly matched answers, count crossings by the end of the matched value. Report unclassified openings separately and publish the full outputs. This check measures how an answer-trained probe behaves at generation time, including halts on words before the disputed detail.

## Development outcome and a second experiment

The frozen primary selection is `recalibrated-v1` at threshold `0.9740611507233753`. On validation it caught 30/96 prior substitutions and flagged 15/192 supported answers. The published threshold caught 34/96 and flagged 21/192, exceeding the cap on two layouts. The two refitted probes each caught 26/96 at their feasible thresholds. Because the selected weights stayed identical and its threshold rose, this primary candidate cannot recover a previously missed case. We will still evaluate and report it under the original rules. [Frozen development results](../artifacts/temporal-v2/dev/selection.json)

Before opening the fresh test, the project registered a separate **source-role probe**. The input is the raw chart and the answer prefix available at the scoring token. Find the longest substantive suffix of up to five words with a unique whole-word match in the chart before the question. Read the Qwen block-12 residual at the end of the matched chart occurrence. Train an L2 logistic classifier (`0.1`) on matched training examples. If the source cannot be uniquely resolved, use the original v1 risk. Select one shared threshold on the same development validation set and with the same per-layout false-alarm constraint. Publish the weights and threshold before the fresh evaluation. Evaluate both matched and fallback rows and record the raw match offsets. Ground-truth values, source offsets and visit order are available for evaluation only. For free generation, apply this scorer to the same recorded prefixes and compare halts by the end of directly matched answers.

A text-only comparison uses the same match and the text on its line before that match. Prior cues are *prior, previous, previously, earlier, older, past, last, historical, archived, closed, preceding,* and *former*. Current cues are *current, today, active, present, latest, now, ongoing, confirmed, this encounter, this visit, most recent,* and *in progress*. Match whole words without regard to case. A prior cue alone flags the candidate, and a current cue alone allows it. Conflicting or absent cues and unresolved matches defer to the published v1 decision. This comparison helps distinguish the value of an internal activation from the information available in explicit temporal wording.

The sampling-domain label was renamed before fresh scoring as part of the repository attribution update. The protocol above records the final seed.

The source-role weights and threshold were frozen at probe SHA-256 `8d0f4e8d359affa07be74cb3a2e8c564296c21f654861068530833f9d436358c` and development-record SHA-256 `174688e627c7e810449a89fbdd402bacb8ea7dde0731d1b130e1d998abf8f818`. In addition to the fixed candidate test, report the chronology-word comparison on the known v1 misses and on the same free-generation prefixes.
