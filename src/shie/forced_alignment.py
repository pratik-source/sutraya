"""
CTC Viterbi forced alignment utilities for SUTRAYA SHIE.

Input log probabilities must have shape [T, C].
Target is a sequence of non-blank vocabulary IDs.
"""
from dataclasses import dataclass
from math import isfinite
from typing import List, Optional, Sequence, Tuple

@dataclass
class CharacterSpan:
    target_index: int
    token_id: int
    start_frame: Optional[int]
    end_frame: Optional[int]
    mean_log_probability: Optional[float]

def viterbi_forced_align(log_probs, target: Sequence[int], blank_id: int = 0) -> Tuple[List[CharacterSpan], List[int], float]:
    """
    Find the highest-scoring CTC path constrained to produce `target`.

    Returns:
      spans: one entry per target token; frame interval is inclusive.
      path: best CTC state path as vocabulary token IDs (including blanks).
      score: summed log probability of the best path.

    Raises ValueError when the target cannot be aligned to the supplied frames.
    """
    import torch

    if not isinstance(log_probs, torch.Tensor):
        log_probs = torch.as_tensor(log_probs)
    if log_probs.ndim != 2:
        raise ValueError(f"log_probs must be [T,C], received {tuple(log_probs.shape)}")

    time_steps, num_classes = log_probs.shape
    target = [int(token) for token in target]

    if time_steps < 1:
        raise ValueError("At least one time step is required.")
    if not target:
        return [], [blank_id] * time_steps, float(log_probs[:, blank_id].sum().item())
    if any(token == blank_id for token in target):
        raise ValueError("Target sequence must not contain the CTC blank token.")
    if any(token < 0 or token >= num_classes for token in target):
        raise ValueError("Target contains a token outside the model class range.")

    minimum_frames = len(target) + sum(
        target[i] == target[i - 1] for i in range(1, len(target))
    )
    if time_steps < minimum_frames:
        raise ValueError(
            f"CTC-infeasible target: {time_steps} frames for minimum {minimum_frames}."
        )

    # CPU Python lists avoid thousands of tiny GPU synchronization operations.
    probabilities = log_probs.detach().float().cpu().tolist()

    # CTC state sequence: blank, char, blank, char, ..., blank.
    extended = [blank_id]
    for token in target:
        extended.extend([token, blank_id])
    state_count = len(extended)
    negative_infinity = float("-inf")

    scores = [[negative_infinity] * state_count for _ in range(time_steps)]
    back = [[-1] * state_count for _ in range(time_steps)]

    scores[0][0] = probabilities[0][blank_id]
    scores[0][1] = probabilities[0][extended[1]]

    for t in range(1, time_steps):
        for state in range(state_count):
            best_score = scores[t - 1][state]
            best_state = state

            if state >= 1 and scores[t - 1][state - 1] > best_score:
                best_score = scores[t - 1][state - 1]
                best_state = state - 1

            if (
                state >= 2
                and extended[state] != blank_id
                and extended[state] != extended[state - 2]
                and scores[t - 1][state - 2] > best_score
            ):
                best_score = scores[t - 1][state - 2]
                best_state = state - 2

            if isfinite(best_score):
                scores[t][state] = best_score + probabilities[t][extended[state]]
                back[t][state] = best_state

    final_options = [
        (scores[-1][state_count - 1], state_count - 1),
        (scores[-1][state_count - 2], state_count - 2),
    ]
    final_score, state = max(final_options, key=lambda item: item[0])
    if not isfinite(final_score):
        raise ValueError("No valid CTC alignment exists for this target.")

    state_path = [state]
    for t in range(time_steps - 1, 0, -1):
        state = back[t][state]
        if state < 0:
            raise RuntimeError("Forced-alignment backtrace encountered an invalid state.")
        state_path.append(state)
    state_path.reverse()

    token_path = [extended[state_id] for state_id in state_path]
    spans = []
    for target_index, token_id in enumerate(target):
        state_id = 2 * target_index + 1
        frames = [t for t, path_state in enumerate(state_path) if path_state == state_id]
        if not frames:
            spans.append(CharacterSpan(target_index, token_id, None, None, None))
            continue
        start_frame, end_frame = min(frames), max(frames)
        mean_logp = sum(probabilities[t][token_id] for t in frames) / len(frames)
        spans.append(CharacterSpan(target_index, token_id, start_frame, end_frame, mean_logp))

    return spans, token_path, float(final_score)
