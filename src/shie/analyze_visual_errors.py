"""
Generate character-level visual error evidence for SHIE.

Run from the SUTRAYA project root:
    python -m src.shie.analyze_visual_errors

Uses the existing best validation checkpoint; does not train or use the IAM test set.
Outputs:
    experiments/shie/visual_error_analysis/
"""
from pathlib import Path
from collections import Counter
import csv
import json

import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

from src.shie.config import load_config, resolve_project_path
from src.shie.dataset import IAMDataset, collate_iam_batch
from src.shie.model import SHIEModel
from src.shie.decoder import CTCDecoder
from src.shie.forced_alignment import viterbi_forced_align


def edit_alignment(reference, prediction):
    """Minimum-edit alignment. Returns (operation, ref_index, pred_index)."""
    n, m = len(reference), len(prediction)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0], back[i][0] = i, "D"
    for j in range(1, m + 1):
        dp[0][j], back[0][j] = j, "I"

    # Deterministic tie-breaking: diagonal, deletion, insertion.
    priority = {"M": 0, "S": 0, "D": 1, "I": 2}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == prediction[j - 1] else 1
            options = [
                (dp[i - 1][j - 1] + cost, "M" if cost == 0 else "S"),
                (dp[i - 1][j] + 1, "D"),
                (dp[i][j - 1] + 1, "I"),
            ]
            dp[i][j], back[i][j] = min(options, key=lambda item: (item[0], priority[item[1]]))

    i, j, operations = n, m, []
    while i or j:
        op = back[i][j]
        if op in ("M", "S"):
            operations.append((op, i - 1, j - 1))
            i -= 1
            j -= 1
        elif op == "D":
            operations.append((op, i - 1, None))
            i -= 1
        else:
            operations.append((op, None, j - 1))
            j -= 1
    return list(reversed(operations))


def token_to_char(idx_to_char, token_id):
    value = idx_to_char.get(str(int(token_id)), idx_to_char.get(int(token_id)))
    if value is None:
        raise KeyError(f"Token ID {token_id} missing from idx_to_char.")
    return value



def save_annotated_line(
    image_tensor,
    spans,
    errors,
    time_steps,
    output_path,
    reference,
    prediction
):
    """Save resized model input with approximate reference-character regions highlighted."""

    # Dataset normalization: dark ink=0, white background=1.
    image_array = (
        (1.0 - image_tensor.detach().cpu().squeeze(0).clamp(0, 1)) * 255
    ).byte().numpy()

    image = Image.fromarray(image_array, mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)

    width, height = image.size
    scale = width / max(time_steps, 1)

    for error in errors:

        ref_index = error["reference_index"]

        # Insertions have no corresponding reference character.
        # Their reference_index is stored as an empty string.
        if ref_index is None or ref_index == "":
            continue

        ref_index = int(ref_index)

        if ref_index < 0 or ref_index >= len(spans):
            continue

        span = spans[ref_index]

        # Skip characters without an aligned frame.
        if span.start_frame is None or span.end_frame is None:
            continue

        x1 = max(0, int(span.start_frame * scale) - 2)
        x2 = min(width - 1, int((span.end_frame + 1) * scale) + 2)

        draw.rectangle(
            (x1, 1, x2, height - 2),
            outline=(255, 0, 0),
            width=2
        )

        label = (
            f'{error["operation"]}: '
            f'{error["reference_char"]}->'
            f'{error["prediction_char"]}'
        )

        draw.text((x1, 1), label, fill=(255, 0, 0))

    image.save(output_path)



def main():
    config = load_config()
    requested_device = config["hardware"]["device"]
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested in SHIE config, but CUDA is unavailable.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataset = IAMDataset(split="validation", config=config)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
        collate_fn=collate_iam_batch,
    )

    model = SHIEModel(
        num_classes=config["model"]["num_classes"],
        config=config,
    ).to(device)

    checkpoint_path = (
        resolve_project_path(config["checkpoint"]["directory"]) / "best_model.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    decoder = CTCDecoder(config["dataset"]["vocabulary_path"])
    blank_id = int(config["training"]["blank_index"])
    idx_to_char = dataset.vocabulary["idx_to_char"]

    output_dir = resolve_project_path("experiments/shie/visual_error_analysis")
    horizontal_stride = int(config["dataset"]["horizontal_stride"])
    crops_dir = output_dir / "annotated_lines"
    crops_dir.mkdir(parents=True, exist_ok=True)

    error_rows = []
    confusion_counts = Counter()
    sample_count = 0
    skipped_alignments = 0

    with torch.inference_mode():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            input_lengths = batch["input_lengths"].cpu()
            logits = model(images, input_lengths=input_lengths)
            log_probs = logits.log_softmax(dim=-1)

            target_chunks = []
            offset = 0
            for target_length in batch["target_lengths"].tolist():
                target_chunks.append(batch["targets"][offset:offset + target_length].tolist())
                offset += target_length

            predictions = decoder.decode_batch(logits, input_lengths=input_lengths)

            for index, target_ids in enumerate(target_chunks):
                sample_count += 1
                sample_id = str(batch["sample_ids"][index])
                reference = str(batch["texts"][index])
                prediction = str(predictions[index])
                time_length = int(input_lengths[index].item())
                sample_log_probs = log_probs[index, :time_length]

                try:
                    spans, _, alignment_score = viterbi_forced_align(
                        sample_log_probs, target_ids, blank_id=blank_id
                    )
                except (ValueError, RuntimeError) as exc:
                    skipped_alignments += 1
                    error_rows.append({
                        "sample_id": sample_id,
                        "reference": reference,
                        "prediction": prediction,
                        "operation": "alignment_failed",
                        "reference_index": "",
                        "prediction_index": "",
                        "reference_char": "",
                        "prediction_char": "",
                        "start_frame": "",
                        "end_frame": "",
                        "mean_target_log_probability": "",
                        "alignment_score": "",
                        "note": str(exc),
                    })
                    continue

                operations = edit_alignment(reference, prediction)
                sample_errors = []
                for op, ref_index, pred_index in operations:
                    if op == "M":
                        continue
                    ref_char = reference[ref_index] if ref_index is not None else ""
                    pred_char = prediction[pred_index] if pred_index is not None else ""
                    if op == "S":
                        confusion_counts[(ref_char, pred_char)] += 1

                    span = spans[ref_index] if ref_index is not None and ref_index < len(spans) else None
                    row = {
                        "sample_id": sample_id,
                        "reference": reference,
                        "prediction": prediction,
                        "operation": {"S": "substitution", "D": "deletion", "I": "insertion"}[op],
                        "reference_index": ref_index if ref_index is not None else "",
                        "prediction_index": pred_index if pred_index is not None else "",
                        "reference_char": ref_char if ref_char else "[none]",
                        "prediction_char": pred_char if pred_char else "[none]",
                        "start_frame": span.start_frame if span and span.start_frame is not None else "",
                        "end_frame": span.end_frame if span and span.end_frame is not None else "",
                        "mean_target_log_probability": span.mean_log_probability if span and span.mean_log_probability is not None else "",
                        "alignment_score": alignment_score,
                        "note": "Frame interval is approximate; CTC alignment is against the reference transcript.",
                    }
                    error_rows.append(row)
                    sample_errors.append(row)

                if sample_errors:
                    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in sample_id)
                    save_annotated_line(
                        batch["images"][index, :, :, :min(
                            batch["images"].shape[-1], time_length * horizontal_stride
                        )],
                        spans, sample_errors, time_length,
                        crops_dir / f"{safe_id}.png", reference, prediction
                    )

            if sample_count % 80 == 0:
                print(f"Processed {sample_count}/{len(dataset)} validation samples")

    if sample_count and sample_count % 80:
        print(f"Processed {sample_count}/{len(dataset)} validation samples")

    csv_path = output_dir / "visual_error_alignment.csv"
    fieldnames = [
        "sample_id", "reference", "prediction", "operation",
        "reference_index", "prediction_index", "reference_char", "prediction_char",
        "start_frame", "end_frame", "mean_target_log_probability",
        "alignment_score", "note"
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(error_rows)

    confusion_path = output_dir / "visual_confusion_pairs.csv"
    with confusion_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reference_character", "predicted_character", "count"])
        for (ref_char, pred_char), count in confusion_counts.most_common():
            writer.writerow([ref_char, pred_char, count])

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "device": str(device),
        "validation_samples_processed": sample_count,
        "alignment_failures": skipped_alignments,
        "substitution_pairs": len(confusion_counts),
        "annotated_line_images": len(list(crops_dir.glob("*.png"))),
        "output_directory": str(output_dir),
        "important_note": (
            "The red rectangles show approximate CTC reference-character spans, not "
            "ground-truth character bounding boxes. Insertions do not have a reference "
            "character span. Alignment confidence is not a calibrated probability."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSHIE visual error analysis completed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
