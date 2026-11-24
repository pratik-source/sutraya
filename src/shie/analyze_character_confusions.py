"""
SHIE character-level confusion analysis.

Run from the SUTRAYA project root:
    python -m src.shie.analyze_character_confusions

Input:
    experiments/shie/validation_profiling/validation_predictions.csv

Outputs:
    experiments/shie/character_analysis/
"""
from pathlib import Path
from collections import Counter
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "experiments/shie/validation_profiling/validation_predictions.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments/shie/character_analysis"

def align_chars(reference: str, prediction: str):
    """Return minimum-edit alignment operations: M, S, D, I."""
    n, m = len(reference), len(prediction)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0], back[i][0] = i, "D"
    for j in range(1, m + 1):
        dp[0][j], back[0][j] = j, "I"

    priority = {"M": 0, "S": 0, "D": 1, "I": 2}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == prediction[j - 1] else 1
            candidates = [
                (dp[i - 1][j - 1] + cost, "M" if cost == 0 else "S"),
                (dp[i - 1][j] + 1, "D"),
                (dp[i][j - 1] + 1, "I"),
            ]
            dp[i][j], back[i][j] = min(
                candidates, key=lambda item: (item[0], priority[item[1]])
            )

    i, j, operations = n, m, []
    while i or j:
        operation = back[i][j]
        if operation in ("M", "S"):
            operations.append((operation, reference[i - 1], prediction[j - 1]))
            i -= 1
            j -= 1
        elif operation == "D":
            operations.append(("D", reference[i - 1], ""))
            i -= 1
        else:
            operations.append(("I", "", prediction[j - 1]))
            j -= 1
    return list(reversed(operations))

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing validation predictions: {INPUT_CSV}")

    data = pd.read_csv(INPUT_CSV)
    required = {"sample_id", "reference", "prediction"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    substitutions, deleted, inserted = Counter(), Counter(), Counter()
    examples = []
    case_insensitive_errors = 0

    for row in data.itertuples(index=False):
        reference = str(row.reference)
        prediction = str(row.prediction)
        for op, ref_char, pred_char in align_chars(reference, prediction):
            if op == "S":
                substitutions[(ref_char, pred_char)] += 1
            elif op == "D":
                deleted[ref_char] += 1
            elif op == "I":
                inserted[pred_char] += 1
            if op != "M":
                examples.append({
                    "sample_id": row.sample_id,
                    "operation": {"S": "substitution", "D": "deletion", "I": "insertion"}[op],
                    "reference_character": ref_char if ref_char else "[none]",
                    "predicted_character": pred_char if pred_char else "[none]",
                    "reference": reference,
                    "prediction": prediction,
                    "cer": getattr(row, "cer", None),
                    "wer": getattr(row, "wer", None),
                })
        case_insensitive_errors += sum(
            op != "M" for op, _, _ in align_chars(reference.lower(), prediction.lower())
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"reference_character": a, "predicted_character": b, "count": count}
        for (a, b), count in substitutions.most_common()
    ]).to_csv(OUTPUT_DIR / "character_confusion_pairs.csv", index=False)

    pd.DataFrame([
        {"deleted_character": char, "count": count}
        for char, count in deleted.most_common()
    ]).to_csv(OUTPUT_DIR / "deleted_characters.csv", index=False)

    pd.DataFrame([
        {"inserted_character": char, "count": count}
        for char, count in inserted.most_common()
    ]).to_csv(OUTPUT_DIR / "inserted_characters.csv", index=False)

    pd.DataFrame(examples).to_csv(OUTPUT_DIR / "character_error_examples.csv", index=False)

    reference_characters = sum(len(str(value)) for value in data["reference"])
    total_errors = sum(substitutions.values()) + sum(deleted.values()) + sum(inserted.values())
    summary = {
        "validation_samples": len(data),
        "substitutions": sum(substitutions.values()),
        "deletions": sum(deleted.values()),
        "insertions": sum(inserted.values()),
        "total_character_errors": total_errors,
        "case_sensitive_cer": total_errors / reference_characters,
        "case_insensitive_character_errors": case_insensitive_errors,
        "case_insensitive_cer": case_insensitive_errors / reference_characters,
        "alignment_rule": "Minimum Levenshtein distance; ties prefer diagonal, then deletion, then insertion.",
        "note": "Substitution/deletion/insertion assignments can depend on tie-breaking; total edit distance is stable."
    }
    (OUTPUT_DIR / "character_analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("SHIE character analysis completed.")
    print(f"Samples: {len(data)}")
    print(f"Character errors: {total_errors}")
    print(f"Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
