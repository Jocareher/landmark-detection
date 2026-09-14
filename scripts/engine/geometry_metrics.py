from __future__ import annotations

import numpy as np


LANDMARK_CONNECTION_GROUPS: list[tuple[tuple[int, ...], bool]] = [
    (tuple(range(0, 17)), False),
    (tuple(range(17, 22)), False),
    (tuple(range(22, 27)), False),
    (tuple(range(27, 31)), False),
    (tuple(range(31, 36)), False),
    (tuple(range(36, 42)), True),
    (tuple(range(42, 48)), True),
    (tuple(range(48, 60)), True),
    (tuple(range(60, 68)), True),
    ((68,), False),
    ((69,), False),
    ((70,), False),
    ((71,), False),
]


def _get_landmark_groups(
    num_landmarks: int,
) -> list[tuple[list[int], bool]]:
    """Return the topology groups truncated to the requested landmark count."""
    groups: list[tuple[list[int], bool]] = []
    for group_indices, close_loop in LANDMARK_CONNECTION_GROUPS:
        indices = [index for index in group_indices if index < num_landmarks]
        if indices:
            groups.append((indices, close_loop))
    return groups


def _build_point_to_line_support_segments(
    num_landmarks: int,
) -> list[list[tuple[int, int]]]:
    """Build local GT support segments per landmark.

    Each landmark is compared against the local GT polyline around it:
    - internal landmarks use the two adjacent segments `(prev, current)` and
      `(current, next)`
    - open-chain endpoints use their single incident segment
    - singleton landmarks fall back to a degenerate point segment

    This definition guarantees zero point-to-line error for a perfect prediction.
    """
    segments: list[list[tuple[int, int]]] = [[] for _ in range(num_landmarks)]
    for indices, close_loop in _get_landmark_groups(num_landmarks):
        if len(indices) == 1:
            index = indices[0]
            segments[index] = [(index, index)]
            continue

        for position, landmark_index in enumerate(indices):
            if close_loop:
                previous_index = indices[(position - 1) % len(indices)]
                next_index = indices[(position + 1) % len(indices)]
                segments[landmark_index] = [
                    (previous_index, landmark_index),
                    (landmark_index, next_index),
                ]
                continue

            previous_index = indices[position - 1] if position > 0 else None
            next_index = indices[position + 1] if position < len(indices) - 1 else None
            if previous_index is None and next_index is None:
                segments[landmark_index] = [(landmark_index, landmark_index)]
            elif previous_index is None:
                segments[landmark_index] = [(landmark_index, next_index)]
            elif next_index is None:
                segments[landmark_index] = [(previous_index, landmark_index)]
            else:
                segments[landmark_index] = [
                    (previous_index, landmark_index),
                    (landmark_index, next_index),
                ]
    return segments


def _point_to_segment_distance(
    point: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Return the Euclidean distance between one point and one line segment."""
    segment_vector = segment_end - segment_start
    segment_length_sq = float(np.dot(segment_vector, segment_vector))
    if segment_length_sq <= eps:
        return float(np.linalg.norm(point - segment_start))

    projection = float(
        np.dot(point - segment_start, segment_vector) / segment_length_sq
    )
    projection = min(1.0, max(0.0, projection))
    closest_point = segment_start + projection * segment_vector
    return float(np.linalg.norm(point - closest_point))


def compute_per_landmark_point_to_point_distances(
    predicted_landmarks: np.ndarray,
    target_landmarks: np.ndarray,
) -> np.ndarray:
    """Compute the per-landmark point-to-point Euclidean distances."""
    return np.linalg.norm(predicted_landmarks - target_landmarks, axis=1).astype(
        np.float32
    )


def compute_per_landmark_point_to_line_distances(
    predicted_landmarks: np.ndarray,
    target_landmarks: np.ndarray,
) -> np.ndarray:
    """Compute the per-landmark point-to-line distances against GT local segments."""
    num_landmarks = int(min(len(predicted_landmarks), len(target_landmarks)))
    support_segments = _build_point_to_line_support_segments(num_landmarks)
    distances = np.zeros(num_landmarks, dtype=np.float32)

    for landmark_index in range(num_landmarks):
        segments = support_segments[landmark_index]
        if not segments:
            distances[landmark_index] = float(
                np.linalg.norm(
                    predicted_landmarks[landmark_index]
                    - target_landmarks[landmark_index]
                )
            )
            continue

        distances[landmark_index] = min(
            _point_to_segment_distance(
                point=predicted_landmarks[landmark_index].astype(np.float32),
                segment_start=target_landmarks[start_index].astype(np.float32),
                segment_end=target_landmarks[end_index].astype(np.float32),
            )
            for start_index, end_index in segments
        )
    return distances
