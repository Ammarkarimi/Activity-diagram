from __future__ import annotations

import re


def normalize(text: str) -> str:

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def defect_key(defect) -> str:

    return "|".join(
        [
            defect.category,
            ",".join(
                sorted(
                    defect.node_ids
                )
            ),
            ",".join(
                sorted(
                    defect.edge_ids
                )
            ),
            ",".join(
                sorted(
                    defect.requirement_ids
                )
            ),
            normalize(
                defect.description
            ),
        ]
    )


def deduplicate_defects(
    defects,
):

    result = []
    seen = set()

    for defect in defects:

        key = defect_key(
            defect
        )

        if key in seen:
            continue

        seen.add(key)

        result.append(
            defect
        )

    return result