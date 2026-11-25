"""Print the held-out results and first threshold crossing from a completed run."""

import argparse
from pathlib import Path

from eval.validate_trace import parse_json, validate_probe, validate_trace


def report(folder: Path) -> str:
    metrics = parse_json((folder / 'metrics.json').read_bytes())
    trace = parse_json((folder / 'trace.json').read_bytes())
    validate_trace(trace)
    validate_probe(trace, folder / 'probe.json')
    if trace['probe_id'] != metrics['probe_id']:
        raise ValueError('Metrics and trace refer to different probes')
    threshold = metrics['threshold']
    test = metrics['splits']['test']
    lines = [
        f"Held-out test: {test['rows']} candidate answers, {test['pairs']} chart pairs, "
        f"{test['groups']} matched groups.",
        '',
        '| Scorer | AUROC | F1 | False alarms | Misses |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for name, key in [('Residual-stream probe', 'test'), ('Whole-prompt word match', 'test_lexical_baseline')]:
        result = metrics[key]
        lines.append(f"| {name} | {result['auroc']:.3f} | {result['f1']:.3f} | "
                     f"{result['fp']}/{result['fp'] + result['tn']} | "
                     f"{result['fn']}/{result['fn'] + result['tp']} |")
    lines.append(f"\nValidation-selected threshold: {threshold:.12g}.")
    tokens = trace['tokens']
    crossing = next((t for t in tokens if t['risk_score'] >= threshold), None)
    lines.append(f"Greedy trace ({len(tokens)} tokens): {''.join(t['text'] for t in tokens)!r}")
    if crossing is not None:
        lines.append(f"First token at or above that threshold: position {crossing['position']}, "
                     f"text {crossing['text']!r}, score {crossing['risk_score']:.12g}.")
    else:
        lines.append('Every token in this trace falls below the validation-selected threshold.')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, nargs='?', default=Path('artifacts/baseline'))
    args = parser.parse_args()
    print(report(args.folder))


if __name__ == '__main__':
    main()
