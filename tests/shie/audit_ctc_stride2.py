
from pathlib import Path
import pandas as pd

ROOT = Path("datasets/iam/annotations")
SPLITS = ["train", "validation", "test"]

def repeated_count(text):
    return sum(
        text[i] == text[i - 1]
        for i in range(1, len(text))
    )

def main():

    total = 0
    feasible = 0
    problematic = []

    for split in SPLITS:

        df = pd.read_csv(ROOT / f"{split}.csv")

        for _, row in df.iterrows():

            text = str(row["text"])

            resized_width = round(
                int(row["width"]) * 32 /
                int(row["height"])
            )

            # Revised horizontal stride = 2
            input_length = resized_width // 2

            required = (
                len(text) + repeated_count(text)
            )

            total += 1

            if input_length >= required:
                feasible += 1
            else:
                problematic.append({
                    "sample": row["sample_id"],
                    "available": input_length,
                    "required": required,
                    "deficit": required - input_length
                })

    print("=" * 50)
    print("SUTRAYA CTC STRIDE-2 AUDIT")
    print("=" * 50)

    print("Total samples:", total)
    print("Feasible samples:", feasible)
    print("Problematic samples:", len(problematic))

    print(
        "Feasibility:",
        round(feasible / total * 100, 4),
        "%"
    )

    for item in problematic:
        print(item)

    print("\nAUDIT COMPLETED")


if __name__ == "__main__":
    main()
