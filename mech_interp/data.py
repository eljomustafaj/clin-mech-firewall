"""Build paired chart-grounding examples using only the checked-in source manifest."""

import argparse
import json
from pathlib import Path
from typing import TypedDict


SOURCE = Path(__file__).resolve().parents[1] / "data" / "synthetic_chart_v1.json"


class Example(TypedDict):
    id: str
    pair_id: str
    group_id: str
    split: str
    prompt: str
    completion: str
    target_start: int
    target_end: int
    target_text: str
    label: int
    source_version: str
    layout_id: str
    field_id: str
    chart: str
    current_value: str
    prior_value: str
    candidate_value: str
    alternative_location: str


def _chart_pair(version, layout, field, group_id, current, alternative, prior):
    values = field["values"]
    chart = layout["chart"].format(
        field=field["field"], current=values[current], prior=values[prior]
    )
    question = field["question"]
    instruction = layout["question"].format(
        question=question, question_lower=question[0].lower() + question[1:]
    )
    prompt = chart + "\n\n" + instruction
    pair_id = f"{group_id}.current-{current}"
    for label, candidate in enumerate((current, alternative)):
        completion = " " + values[candidate] + "."
        target = field["targets"][candidate]
        start = completion.index(target)
        yield Example(
            id=f"{pair_id}.candidate-{candidate}",
            pair_id=pair_id,
            group_id=group_id,
            split=layout["split"],
            prompt=prompt,
            completion=completion,
            target_start=start,
            target_end=start + len(target),
            target_text=target,
            label=label,
            source_version=version,
            layout_id=layout["id"],
            field_id=field["id"],
            chart=chart,
            current_value=values[current],
            prior_value=values[prior],
            candidate_value=values[candidate],
            alternative_location="prior" if prior == alternative else "absent",
        )


def build_dataset() -> list[Example]:
    """Return deterministic examples. A label of 1 means unsupported at this visit."""
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    examples = []
    for layout_index, layout in enumerate(source["layouts"]):
        for field_index, field in enumerate(source["fields"]):
            for match_index, (left, right) in enumerate(layout["matchings"]):
                # Half of the quartets require distinguishing prior from current.
                reuse_alternative = (layout_index + field_index + match_index) % 2 == 0
                other_values = [i for i in range(len(field["values"])) if i not in (left, right)]
                group_id = f"{layout['id']}.{field['id']}.{left}-{right}"
                for position, (current, alternative) in enumerate(((left, right), (right, left))):
                    prior = alternative if reuse_alternative else other_values[position]
                    examples.extend(_chart_pair(
                        source["version"], layout, field, group_id, current, alternative, prior
                    ))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON Lines file")
    args = parser.parse_args()
    examples = build_dataset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as out:
        for example in examples:
            out.write(json.dumps(example, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
