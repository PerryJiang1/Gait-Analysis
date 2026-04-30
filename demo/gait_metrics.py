import json
import os
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter


# COCO 17 keypoint names
JOINT_NAMES = [
    "nose",          # 0
    "left_eye",      # 1
    "right_eye",     # 2
    "left_ear",      # 3
    "right_ear",     # 4
    "left_shoulder", # 5
    "right_shoulder",# 6
    "left_elbow",    # 7
    "right_elbow",   # 8
    "left_wrist",    # 9
    "right_wrist",   # 10
    "left_hip",      # 11
    "right_hip",     # 12
    "left_knee",     # 13
    "right_knee",    # 14
    "left_ankle",    # 15
    "right_ankle"    # 16
]

JOINT_IDX = {name: i for i, name in enumerate(JOINT_NAMES)}


def load_track_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = []
    keypoints_all = []

    for item in data:
        if item.get("person_idx") is None:
            continue
        if "keypoints" not in item:
            continue

        frames.append(item["frame"])
        keypoints_all.append(np.array(item["keypoints"], dtype=np.float32))  # shape (17,2)

    if len(keypoints_all) == 0:
        raise ValueError("No valid tracked keypoints found in track.json")

    frames = np.array(frames, dtype=np.int32)
    keypoints_all = np.stack(keypoints_all, axis=0)  # shape (T,17,2)
    return frames, keypoints_all

def crop_frame_range(frames, keypoints_all, start_frame=None, end_frame=None):
    """
    Keep only frames in [start_frame, end_frame].
    If start_frame or end_frame is None, keep that side open.
    """
    mask = np.ones(len(frames), dtype=bool)

    if start_frame is not None:
        mask &= (frames >= start_frame)

    if end_frame is not None:
        mask &= (frames <= end_frame)

    cropped_frames = frames[mask]
    cropped_keypoints_all = keypoints_all[mask]

    if len(cropped_frames) == 0:
        raise ValueError("No frames left after cropping. Check start_frame/end_frame.")

    return cropped_frames, cropped_keypoints_all


def compute_joint_change(y_values):
    y_values = np.asarray(y_values)
    value_range = float(np.max(y_values) - np.min(y_values))
    value_std = float(np.std(y_values))
    total_motion = float(np.sum(np.abs(np.diff(y_values))))
    return value_range, value_std, total_motion


def plot_all_joints_time_vs_y(frames, keypoints_all, save_path=None):
    T, J, D = keypoints_all.shape
    assert J == 17 and D == 2, f"Expected shape (T,17,2), got {keypoints_all.shape}"

    y_all = keypoints_all[:, :, 1]

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in range(17)]

    plt.figure(figsize=(16, 8))

    stats = []

    for j in range(17):
        y = y_all[:, j]
        plt.plot(frames, y, label=f"{j}: {JOINT_NAMES[j]}", color=colors[j], linewidth=2)

        value_range, value_std, total_motion = compute_joint_change(y)
        stats.append({
            "joint_id": j,
            "joint_name": JOINT_NAMES[j],
            "range": value_range,
            "std": value_std,
            "total_motion": total_motion
        })

    plt.xlabel("Time (frame)", fontsize=12)
    plt.ylabel("Y coordinate", fontsize=12)
    plt.title("17 Keypoints: Time vs Y Coordinate", fontsize=14)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.gca().invert_yaxis()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()

    return stats


def print_top_joints(stats, top_k=5, metric="range"):
    sorted_stats = sorted(stats, key=lambda x: x[metric], reverse=True)

    print(f"\nTop {top_k} joints by {metric}:")
    for i, s in enumerate(sorted_stats[:top_k], 1):
        print(
            f"{i}. joint {s['joint_id']:2d} ({s['joint_name']:15s}) | "
            f"range={s['range']:.2f}, std={s['std']:.2f}, total_motion={s['total_motion']:.2f}"
        )

    print(f"\nLargest change by {metric}: "
          f"joint {sorted_stats[0]['joint_id']} ({sorted_stats[0]['joint_name']})")


def interpolate_nan(y):
    y = np.asarray(y, dtype=np.float32).copy()
    idx = np.arange(len(y))
    valid = np.isfinite(y)
    if valid.sum() == 0:
        return y
    if valid.sum() == 1:
        y[~valid] = y[valid][0]
        return y
    y[~valid] = np.interp(idx[~valid], idx[valid], y[valid])
    return y


def smooth_signal(y, window_length=11, polyorder=2):
    y = interpolate_nan(y)

    if len(y) < 5:
        return y

    if window_length >= len(y):
        window_length = len(y) - 1 if len(y) % 2 == 0 else len(y)

    if window_length < 3:
        return y

    if window_length % 2 == 0:
        window_length -= 1

    if window_length <= polyorder:
        return y

    return savgol_filter(y, window_length=window_length, polyorder=polyorder)


def compute_mid_shoulder_mid_hip(keypoints_all):
    left_shoulder = keypoints_all[:, JOINT_IDX["left_shoulder"], :]
    right_shoulder = keypoints_all[:, JOINT_IDX["right_shoulder"], :]
    left_hip = keypoints_all[:, JOINT_IDX["left_hip"], :]
    right_hip = keypoints_all[:, JOINT_IDX["right_hip"], :]

    mid_shoulder = 0.5 * (left_shoulder + right_shoulder)
    mid_hip = 0.5 * (left_hip + right_hip)

    return mid_shoulder, mid_hip


def compute_normalized_joint_y_series(keypoints_all, joint_name):
    """
    normalized y:
        (y_joint - y_midhip) / (y_midshoulder - y_midhip)

    shoulder -> 1
    hip -> 0
    """
    joint_idx = JOINT_IDX[joint_name]
    joint_xy = keypoints_all[:, joint_idx, :]   # (T,2)

    mid_shoulder, mid_hip = compute_mid_shoulder_mid_hip(keypoints_all)

    y_joint = joint_xy[:, 1]
    y_hip = mid_hip[:, 1]
    y_shoulder = mid_shoulder[:, 1]

    denom = y_shoulder - y_hip
    denom = np.where(np.abs(denom) < 1e-6, np.nan, denom)

    y_norm = (y_joint - y_hip) / denom
    return y_norm


def detect_stride_events(y_signal, min_distance_frames=10, prominence=0.03):
    """
    Find local minima by inverting the curve.
    """
    y_smooth = smooth_signal(y_signal, window_length=11, polyorder=2)
    peaks, props = find_peaks(-y_smooth, distance=min_distance_frames, prominence=prominence)
    return y_smooth, peaks, props



def compute_stride_metrics(frames, y_smooth, minima_idx, fps=30.0):
    """
    stride time: distance between consecutive minima
    stride height: max - min within one stride segment
    """
    stride_times = []
    stride_heights = []
    stride_segments = []

    for i in range(len(minima_idx) - 1):
        s = minima_idx[i]
        e = minima_idx[i + 1]

        if e <= s:
            continue

        seg = y_smooth[s:e + 1]
        stride_time = (frames[e] - frames[s]) / fps
        stride_height = float(np.max(seg) - np.min(seg))

        stride_times.append(stride_time)
        stride_heights.append(stride_height)
        stride_segments.append((s, e))

    stride_times = np.array(stride_times, dtype=np.float32)
    stride_heights = np.array(stride_heights, dtype=np.float32)

    return stride_times, stride_heights, stride_segments


def compute_stride_metrics_from_events(frames, signal_smooth, event_idx, fps=30.0):
    """
    stride time: distance between consecutive same-type events
    stride height: max - min within one same-type event segment
    """
    stride_times = []
    stride_heights = []
    stride_segments = []

    event_idx = np.asarray(event_idx, dtype=np.int32)

    for i in range(len(event_idx) - 1):
        s = event_idx[i]
        e = event_idx[i + 1]

        if e <= s:
            continue

        seg = signal_smooth[s:e + 1]
        stride_time = (frames[e] - frames[s]) / fps
        stride_height = float(np.max(seg) - np.min(seg))

        stride_times.append(stride_time)
        stride_heights.append(stride_height)
        stride_segments.append((s, e))

    return (
        np.array(stride_times, dtype=np.float32),
        np.array(stride_heights, dtype=np.float32),
        stride_segments
    )


def find_zero_crossings(frames, y_signal, merge_window_frames=None):
    """
    Return zero-crossing locations as interpolated sample indices and frame values.
    """
    frames = np.asarray(frames, dtype=np.float32)
    y_signal = np.asarray(y_signal, dtype=np.float32)

    zero_indices = []
    zero_frames = []

    def add_zero(z_idx, z_frame):
        if len(zero_indices) > 0 and abs(z_idx - zero_indices[-1]) < 1e-6:
            return
        zero_indices.append(float(z_idx))
        zero_frames.append(float(z_frame))

    for i in range(len(y_signal) - 1):
        y0 = y_signal[i]
        y1 = y_signal[i + 1]

        if not np.isfinite(y0) or not np.isfinite(y1):
            continue

        if abs(y0) < 1e-8:
            add_zero(i, frames[i])

        if y0 * y1 < 0:
            frac = -y0 / (y1 - y0)
            z_idx = i + frac
            z_frame = frames[i] + frac * (frames[i + 1] - frames[i])
            add_zero(z_idx, z_frame)

    if len(y_signal) > 0 and np.isfinite(y_signal[-1]) and abs(y_signal[-1]) < 1e-8:
        add_zero(len(y_signal) - 1, frames[-1])

    zero_indices = np.array(zero_indices, dtype=np.float32)
    zero_frames = np.array(zero_frames, dtype=np.float32)

    if merge_window_frames is None or len(zero_frames) <= 1:
        return zero_indices, zero_frames

    merged_indices = []
    merged_frames = []
    cluster_indices = [zero_indices[0]]
    cluster_frames = [zero_frames[0]]

    for z_idx, z_frame in zip(zero_indices[1:], zero_frames[1:]):
        if z_frame - cluster_frames[-1] <= merge_window_frames:
            cluster_indices.append(z_idx)
            cluster_frames.append(z_frame)
        else:
            merged_indices.append(float(np.mean(cluster_indices)))
            merged_frames.append(float(np.mean(cluster_frames)))
            cluster_indices = [z_idx]
            cluster_frames = [z_frame]

    merged_indices.append(float(np.mean(cluster_indices)))
    merged_frames.append(float(np.mean(cluster_frames)))

    return np.array(merged_indices, dtype=np.float32), np.array(merged_frames, dtype=np.float32)


def compute_curve_integrals(frames, y_signal, fps=30.0):
    frames = np.asarray(frames, dtype=np.float32)
    y_signal = np.asarray(y_signal, dtype=np.float32)
    valid = np.isfinite(frames) & np.isfinite(y_signal)

    if valid.sum() < 2:
        return {
            "signed_area_frames": np.nan,
            "absolute_area_frames": np.nan,
            "signed_area_seconds": np.nan,
            "absolute_area_seconds": np.nan,
        }

    x = frames[valid]
    y = y_signal[valid]
    signed_area_frames = float(np.trapezoid(y, x=x))
    absolute_area_frames = float(np.trapezoid(np.abs(y), x=x))

    return {
        "signed_area_frames": signed_area_frames,
        "absolute_area_frames": absolute_area_frames,
        "signed_area_seconds": signed_area_frames / fps,
        "absolute_area_seconds": absolute_area_frames / fps,
    }


def compute_segment_absolute_area(frames, y_signal, left_zero_frame, right_zero_frame, fps=30.0):
    frames = np.asarray(frames, dtype=np.float32)
    y_signal = np.asarray(y_signal, dtype=np.float32)

    inside = (frames > left_zero_frame) & (frames < right_zero_frame) & np.isfinite(y_signal)
    x = np.concatenate((
        np.array([left_zero_frame], dtype=np.float32),
        frames[inside],
        np.array([right_zero_frame], dtype=np.float32)
    ))
    y = np.concatenate((
        np.array([0.0], dtype=np.float32),
        y_signal[inside],
        np.array([0.0], dtype=np.float32)
    ))

    if len(x) < 2:
        area_frames = np.nan
    else:
        order = np.argsort(x)
        area_frames = float(np.trapezoid(np.abs(y[order]), x=x[order]))

    return area_frames, area_frames / fps if np.isfinite(area_frames) else np.nan


def compute_zero_crossing_step_metrics(frames,
                                       y_diff_smooth,
                                       pos_peaks,
                                       neg_peaks,
                                       fps=30.0):
    """
    Step definition:
      left step: one positive lobe between two zero-crossings
      right step: one negative lobe between two zero-crossings

    Step time is measured in frames between those two zero-crossings.
    Step height is the strongest peak magnitude within that lobe.
    If multiple peaks are inside one lobe, keep only the strongest one.
    Opposite-sign peaks inside a lobe are skipped.
    """
    raw_zero_indices, raw_zero_frames = find_zero_crossings(frames, y_diff_smooth)
    zero_indices, zero_frames = find_zero_crossings(
        frames,
        y_diff_smooth,
        merge_window_frames= int(0.2 * fps)
    )

    pos_peaks = np.asarray(pos_peaks, dtype=np.int32)
    neg_peaks = np.asarray(neg_peaks, dtype=np.int32)

    left = {
        "step_times_frames": [],
        "step_heights": [],
        "step_areas_frames": [],
        "step_areas_seconds": [],
        "step_segments": [],
        "valid_peaks": [],
        "skipped_peaks": [],
    }
    right = {
        "step_times_frames": [],
        "step_heights": [],
        "step_areas_frames": [],
        "step_areas_seconds": [],
        "step_segments": [],
        "valid_peaks": [],
        "skipped_peaks": [],
    }
    assigned_pos = set()
    assigned_neg = set()
    effective_zero_indices = []
    effective_zero_frames = []

    for i in range(len(zero_indices) - 1):
        left_zero_idx = zero_indices[i]
        right_zero_idx = zero_indices[i + 1]
        left_zero_frame = zero_frames[i]
        right_zero_frame = zero_frames[i + 1]
        if right_zero_frame <= left_zero_frame:
            continue

        pos_in_segment = pos_peaks[
            (pos_peaks > left_zero_idx) &
            (pos_peaks < right_zero_idx)
        ]
        neg_in_segment = neg_peaks[
            (neg_peaks > left_zero_idx) &
            (neg_peaks < right_zero_idx)
        ]

        seg_start_idx = max(0, int(np.floor(left_zero_idx)) + 1)
        seg_end_idx = min(len(y_diff_smooth), int(np.ceil(right_zero_idx)))
        segment_values = y_diff_smooth[seg_start_idx:seg_end_idx]
        if len(segment_values) == 0 or not np.any(np.isfinite(segment_values)):
            continue
        segment_sign = 1 if np.nanmean(segment_values) >= 0 else -1
        step_time = float(right_zero_frame - left_zero_frame)

        if segment_sign > 0:
            if len(pos_in_segment) == 0:
                for peak_idx in neg_in_segment:
                    right["skipped_peaks"].append(int(peak_idx))
                    assigned_neg.add(int(peak_idx))
                continue

            keep_peak = int(pos_in_segment[np.argmax(y_diff_smooth[pos_in_segment])])
            step_height = float(y_diff_smooth[keep_peak])
            if np.isfinite(step_time) and np.isfinite(step_height):
                area_frames, area_seconds = compute_segment_absolute_area(
                    frames,
                    y_diff_smooth,
                    left_zero_frame,
                    right_zero_frame,
                    fps=fps
                )
                left["step_times_frames"].append(step_time)
                left["step_heights"].append(step_height)
                left["step_areas_frames"].append(area_frames)
                left["step_areas_seconds"].append(area_seconds)
                left["step_segments"].append((float(left_zero_frame), float(right_zero_frame)))
                left["valid_peaks"].append(keep_peak)
                assigned_pos.add(keep_peak)
                effective_zero_indices.extend([float(left_zero_idx), float(right_zero_idx)])
                effective_zero_frames.extend([float(left_zero_frame), float(right_zero_frame)])

            for peak_idx in pos_in_segment:
                if int(peak_idx) != keep_peak:
                    left["skipped_peaks"].append(int(peak_idx))
                assigned_pos.add(int(peak_idx))
            for peak_idx in neg_in_segment:
                right["skipped_peaks"].append(int(peak_idx))
                assigned_neg.add(int(peak_idx))
        else:
            if len(neg_in_segment) == 0:
                for peak_idx in pos_in_segment:
                    left["skipped_peaks"].append(int(peak_idx))
                    assigned_pos.add(int(peak_idx))
                continue

            keep_peak = int(neg_in_segment[np.argmin(y_diff_smooth[neg_in_segment])])
            step_height = float(-y_diff_smooth[keep_peak])
            if np.isfinite(step_time) and np.isfinite(step_height):
                area_frames, area_seconds = compute_segment_absolute_area(
                    frames,
                    y_diff_smooth,
                    left_zero_frame,
                    right_zero_frame,
                    fps=fps
                )
                right["step_times_frames"].append(step_time)
                right["step_heights"].append(step_height)
                right["step_areas_frames"].append(area_frames)
                right["step_areas_seconds"].append(area_seconds)
                right["step_segments"].append((float(left_zero_frame), float(right_zero_frame)))
                right["valid_peaks"].append(keep_peak)
                assigned_neg.add(keep_peak)
                effective_zero_indices.extend([float(left_zero_idx), float(right_zero_idx)])
                effective_zero_frames.extend([float(left_zero_frame), float(right_zero_frame)])

            for peak_idx in neg_in_segment:
                if int(peak_idx) != keep_peak:
                    right["skipped_peaks"].append(int(peak_idx))
                assigned_neg.add(int(peak_idx))
            for peak_idx in pos_in_segment:
                left["skipped_peaks"].append(int(peak_idx))
                assigned_pos.add(int(peak_idx))

    for peak_idx in pos_peaks:
        if int(peak_idx) not in assigned_pos:
            left["skipped_peaks"].append(int(peak_idx))
    for peak_idx in neg_peaks:
        if int(peak_idx) not in assigned_neg:
            right["skipped_peaks"].append(int(peak_idx))

    for metrics in (left, right):
        metrics["step_times_frames"] = np.array(metrics["step_times_frames"], dtype=np.float32)
        metrics["step_heights"] = np.array(metrics["step_heights"], dtype=np.float32)
        metrics["step_areas_frames"] = np.array(metrics["step_areas_frames"], dtype=np.float32)
        metrics["step_areas_seconds"] = np.array(metrics["step_areas_seconds"], dtype=np.float32)
        metrics["valid_peaks"] = np.array(metrics["valid_peaks"], dtype=np.int32)
        metrics["skipped_peaks"] = np.array(sorted(set(metrics["skipped_peaks"])), dtype=np.int32)

    if len(effective_zero_frames) > 0:
        effective_pairs = sorted(zip(effective_zero_indices, effective_zero_frames), key=lambda x: x[1])
        dedup_effective_indices = []
        dedup_effective_frames = []
        for z_idx, z_frame in effective_pairs:
            if len(dedup_effective_frames) > 0 and abs(z_frame - dedup_effective_frames[-1]) < 1e-6:
                continue
            dedup_effective_indices.append(z_idx)
            dedup_effective_frames.append(z_frame)
        effective_zero_indices = dedup_effective_indices
        effective_zero_frames = dedup_effective_frames

    return {
        "raw_zero_indices": raw_zero_indices,
        "raw_zero_frames": raw_zero_frames,
        "zero_indices": zero_indices,
        "zero_frames": zero_frames,
        "effective_zero_indices": np.array(effective_zero_indices, dtype=np.float32),
        "effective_zero_frames": np.array(effective_zero_frames, dtype=np.float32),
        "left": left,
        "right": right,
    }


def summarize_metric(values, name):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        print(f"{name}: no valid values")
        return

    mean_v = float(np.mean(values))
    std_v = float(np.std(values))
    cv_v = float(std_v / mean_v) if abs(mean_v) > 1e-8 else np.nan

    print(f"\n{name}:")
    print(f"  count = {len(values)}")
    print(f"  mean  = {mean_v:.4f}")
    print(f"  std   = {std_v:.4f}")
    print(f"  cv    = {cv_v:.4f}")


def format_array(values):
    values = np.asarray(values)
    return np.array2string(values, precision=6, separator=", ")


def metric_summary_lines(values, name):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return [f"{name}: no valid values"]

    mean_v = float(np.mean(values))
    std_v = float(np.std(values))
    cv_v = float(std_v / mean_v) if abs(mean_v) > 1e-8 else np.nan

    return [
        f"{name}:",
        f"  count = {len(values)}",
        f"  mean  = {mean_v:.6f}",
        f"  std   = {std_v:.6f}",
        f"  cv    = {cv_v:.6f}",
    ]


def write_metrics_report(save_path, fb_results, wrist_results=None):
    lines = []

    lines.extend([
        "Front/back stride candidates from feet Y difference:",
        f"  bias correction: {fb_results.get('bias_correct')}, bias = {fb_results.get('y_diff_bias', np.nan):.6f}",
        f"  positive peaks: {format_array(fb_results.get('positive_peak_frames', fb_results.get('pos_peaks', [])))}",
        f"  negative peaks: {format_array(fb_results.get('negative_peak_frames', fb_results.get('neg_peaks', [])))}",
        "",
        "Zero-crossing step metrics:",
        f"  raw zero crossings: {format_array(fb_results.get('raw_zero_frames', []))}",
        f"  merged zero crossings: {format_array(fb_results.get('zero_frames', []))}",
        f"  effective zero crossings: {format_array(fb_results.get('effective_zero_frames', []))}",
        f"  valid left peaks: {format_array(fb_results.get('left_valid_peak_frames', fb_results.get('left_valid_peak_idx', [])))}",
        f"  skipped left peaks: {format_array(fb_results.get('left_skipped_peak_frames', fb_results.get('left_skipped_peak_idx', [])))}",
        f"  valid right peaks: {format_array(fb_results.get('right_valid_peak_frames', fb_results.get('right_valid_peak_idx', [])))}",
        f"  skipped right peaks: {format_array(fb_results.get('right_skipped_peak_frames', fb_results.get('right_skipped_peak_idx', [])))}",
        "",
        "Y-diff curve integrals:",
        f"  signed integral: {fb_results.get('curve_signed_integral', np.nan):.6f}",
        f"  absolute integral: {fb_results.get('curve_absolute_integral', np.nan):.6f}",
        f"  left step integrals: {format_array(fb_results.get('left_step_integrals', []))}",
        f"  right step integrals: {format_array(fb_results.get('right_step_integrals', []))}",
    ])

    lines.extend(metric_summary_lines(fb_results.get("left_step_integrals", []), "Left step integral"))
    lines.extend(metric_summary_lines(fb_results.get("right_step_integrals", []), "Right step integral"))
    lines.extend([
        f"  left total integral = {fb_results.get('left_step_integral_total', np.nan):.6f}",
        f"  right total integral = {fb_results.get('right_step_integral_total', np.nan):.6f}",
        "",
    ])

    lines.extend(metric_summary_lines(fb_results.get("left_step_times", []), "Left step time (sec)"))
    lines.extend(metric_summary_lines(fb_results.get("right_step_times", []), "Right step time (sec)"))
    lines.extend(metric_summary_lines(fb_results.get("left_step_heights", []), "Left step height"))
    lines.extend(metric_summary_lines(fb_results.get("right_step_heights", []), "Right step height"))

    n_time = min(len(fb_results.get("left_step_times", [])), len(fb_results.get("right_step_times", [])))
    if n_time > 0:
        time_diff = fb_results["left_step_times"][:n_time] - fb_results["right_step_times"][:n_time]
        lines.extend(metric_summary_lines(np.abs(time_diff), "Absolute left-right step time difference (sec)"))

    n_height = min(len(fb_results.get("left_step_heights", [])), len(fb_results.get("right_step_heights", [])))
    if n_height > 0:
        height_diff = fb_results["left_step_heights"][:n_height] - fb_results["right_step_heights"][:n_height]
        lines.extend(metric_summary_lines(np.abs(height_diff), "Absolute left-right step height difference"))

    if wrist_results is not None:
        left_wrist_smooth, right_wrist_smooth, wrist_diff = wrist_results
        lines.append("")
        lines.extend(metric_summary_lines(left_wrist_smooth, "Left wrist normalized height"))
        lines.extend(metric_summary_lines(right_wrist_smooth, "Right wrist normalized height"))
        lines.extend(metric_summary_lines(np.abs(wrist_diff), "Wrist asymmetry |L-R|"))

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved metrics report to: {save_path}")


def plot_left_right_ankle_waves(frames,
                                left_raw, left_smooth, left_minima,
                                right_raw, right_smooth, right_minima,
                                save_path=None):
    plt.figure(figsize=(14, 6))

    plt.plot(frames, left_raw, label="left ankle raw", alpha=0.35)
    plt.plot(frames, left_smooth, label="left ankle smooth", linewidth=2)

    plt.plot(frames, right_raw, label="right ankle raw", alpha=0.35)
    plt.plot(frames, right_smooth, label="right ankle smooth", linewidth=2)

    if len(left_minima) > 0:
        plt.scatter(frames[left_minima], left_smooth[left_minima],
                    marker="x", s=80, label="left minima")

    if len(right_minima) > 0:
        plt.scatter(frames[right_minima], right_smooth[right_minima],
                    marker="x", s=80, label="right minima")

    plt.xlabel("Time (frame)")
    plt.ylabel("Normalized Y")
    plt.title("Left and Right Ankle Normalized Waves with Detected Stride Events")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


def plot_left_right_stride_metrics(left_stride_times, right_stride_times,
                                   left_stride_heights, right_stride_heights,
                                   save_prefix=None):
    # stride time
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(left_stride_times)), left_stride_times, marker="o", label="left")
    plt.plot(np.arange(len(right_stride_times)), right_stride_times, marker="o", label="right")
    plt.xlabel("Stride index")
    plt.ylabel("Stride time (sec)")
    plt.title("Left vs Right Stride Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_prefix is not None:
        plt.savefig(f"{save_prefix}_left_right_stride_time.png", dpi=200, bbox_inches="tight")
    plt.show()

    # stride height
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(left_stride_heights)), left_stride_heights, marker="o", label="left")
    plt.plot(np.arange(len(right_stride_heights)), right_stride_heights, marker="o", label="right")
    plt.xlabel("Stride index")
    plt.ylabel("Stride height")
    plt.title("Left vs Right Stride Height")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_prefix is not None:
        plt.savefig(f"{save_prefix}_left_right_stride_height.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_left_right_gait_scatter(left_stride_times, right_stride_times,
                                 left_stride_heights, right_stride_heights,
                                 save_path=None):
    plt.figure(figsize=(6, 6))

    plt.scatter(left_stride_times, left_stride_heights, label="left ankle")
    plt.scatter(right_stride_times, right_stride_heights, label="right ankle")

    plt.xlabel("Stride time (sec)")
    plt.ylabel("Stride height")
    plt.title("Gait Scatter: Left vs Right Ankle")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


def plot_left_right_step_metrics(left_step_times, right_step_times,
                                 left_step_heights, right_step_heights,
                                 save_prefix=None):
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(left_step_times)), left_step_times, marker="o", label="left")
    plt.plot(np.arange(len(right_step_times)), right_step_times, marker="o", label="right")
    plt.xlabel("Step index")
    plt.ylabel("Step time (sec)")
    plt.title("Left vs Right Step Time from Zero Crossings")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_prefix is not None:
        plt.savefig(f"{save_prefix}_left_right_step_time.png", dpi=200, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(left_step_heights)), left_step_heights, marker="o", label="left")
    plt.plot(np.arange(len(right_step_heights)), right_step_heights, marker="o", label="right")
    plt.xlabel("Step index")
    plt.ylabel("Step height")
    plt.title("Left vs Right Step Height from Peak Magnitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_prefix is not None:
        plt.savefig(f"{save_prefix}_left_right_step_height.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_left_right_step_scatter(left_step_times, right_step_times,
                                 left_step_heights, right_step_heights,
                                 save_path=None):
    plt.figure(figsize=(6, 6))

    plt.scatter(left_step_times, left_step_heights, label="left step")
    plt.scatter(right_step_times, right_step_heights, label="right step")

    plt.xlabel("Step time (sec)")
    plt.ylabel("Step height")
    plt.title("Step Scatter from Zero-Crossing Windows")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


def compare_left_right_stride_metrics(left_stride_times, right_stride_times,
                                      left_stride_heights, right_stride_heights):
    summarize_metric(left_stride_times, "Left stride time")
    summarize_metric(right_stride_times, "Right stride time")

    summarize_metric(left_stride_heights, "Left stride height")
    summarize_metric(right_stride_heights, "Right stride height")

    n_time = min(len(left_stride_times), len(right_stride_times))
    n_height = min(len(left_stride_heights), len(right_stride_heights))

    if n_time > 0:
        time_diff = left_stride_times[:n_time] - right_stride_times[:n_time]
        summarize_metric(np.abs(time_diff), "Absolute left-right stride time difference")

    if n_height > 0:
        height_diff = left_stride_heights[:n_height] - right_stride_heights[:n_height]
        summarize_metric(np.abs(height_diff), "Absolute left-right stride height difference")


def compare_left_right_step_metrics(left_step_times, right_step_times,
                                    left_step_heights, right_step_heights):
    summarize_metric(left_step_times, "Left step time (sec)")
    summarize_metric(right_step_times, "Right step time (sec)")

    summarize_metric(left_step_heights, "Left step height")
    summarize_metric(right_step_heights, "Right step height")

    n_time = min(len(left_step_times), len(right_step_times))
    n_height = min(len(left_step_heights), len(right_step_heights))

    if n_time > 0:
        time_diff = left_step_times[:n_time] - right_step_times[:n_time]
        summarize_metric(np.abs(time_diff), "Absolute left-right step time difference (sec)")

    if n_height > 0:
        height_diff = left_step_heights[:n_height] - right_step_heights[:n_height]
        summarize_metric(np.abs(height_diff), "Absolute left-right step height difference")


def analyze_wrist_relative_height(frames, keypoints_all, save_path=None):
    left_wrist_norm = compute_normalized_joint_y_series(keypoints_all, "left_wrist")
    right_wrist_norm = compute_normalized_joint_y_series(keypoints_all, "right_wrist")

    left_wrist_smooth = smooth_signal(left_wrist_norm)
    right_wrist_smooth = smooth_signal(right_wrist_norm)

    wrist_diff = left_wrist_smooth - right_wrist_smooth

    plt.figure(figsize=(14, 5))
    plt.plot(frames, left_wrist_smooth, label="left_wrist_norm")
    plt.plot(frames, right_wrist_smooth, label="right_wrist_norm")
    plt.plot(frames, wrist_diff, label="left-right diff", linestyle="--")
    plt.xlabel("Time (frame)")
    plt.ylabel("Normalized Y")
    plt.title("Wrist Relative Height")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()

    summarize_metric(left_wrist_smooth, "Left wrist normalized height")
    summarize_metric(right_wrist_smooth, "Right wrist normalized height")
    summarize_metric(np.abs(wrist_diff), "Wrist asymmetry |L-R|")

    return left_wrist_smooth, right_wrist_smooth, wrist_diff


def analyze_ankle_stride(frames, keypoints_all, fps=30.0, save_prefix=None):
    left_ankle_norm = compute_normalized_joint_y_series(keypoints_all, "left_ankle")
    right_ankle_norm = compute_normalized_joint_y_series(keypoints_all, "right_ankle")

    left_ankle_smooth, left_minima_idx, _ = detect_stride_events(
        left_ankle_norm,
        min_distance_frames=max(8, int(0.5 * fps)),
        prominence=0.03
    )

    right_ankle_smooth, right_minima_idx, _ = detect_stride_events(
        right_ankle_norm,
        min_distance_frames=max(8, int(0.5 * fps)),
        prominence=0.03
    )

    left_stride_times, left_stride_heights, left_stride_segments = compute_stride_metrics(
        frames,
        left_ankle_smooth,
        left_minima_idx,
        fps=fps
    )

    right_stride_times, right_stride_heights, right_stride_segments = compute_stride_metrics(
        frames,
        right_ankle_smooth,
        right_minima_idx,
        fps=fps
    )

    plot_left_right_ankle_waves(
        frames,
        left_ankle_norm, left_ankle_smooth, left_minima_idx,
        right_ankle_norm, right_ankle_smooth, right_minima_idx,
        save_path=None if save_prefix is None else f"{save_prefix}_ankle_waves.png"
    )

    plot_left_right_stride_metrics(
        left_stride_times, right_stride_times,
        left_stride_heights, right_stride_heights,
        save_prefix=save_prefix
    )

    plot_left_right_gait_scatter(
        left_stride_times, right_stride_times,
        left_stride_heights, right_stride_heights,
        save_path=None if save_prefix is None else f"{save_prefix}_scatter.png"
    )

    compare_left_right_stride_metrics(
        left_stride_times, right_stride_times,
        left_stride_heights, right_stride_heights
    )

    return {
        "left_ankle_norm": left_ankle_norm,
        "right_ankle_norm": right_ankle_norm,
        "left_ankle_smooth": left_ankle_smooth,
        "right_ankle_smooth": right_ankle_smooth,
        "left_minima_idx": left_minima_idx,
        "right_minima_idx": right_minima_idx,
        "left_stride_times": left_stride_times,
        "right_stride_times": right_stride_times,
        "left_stride_heights": left_stride_heights,
        "right_stride_heights": right_stride_heights,
        "left_stride_segments": left_stride_segments,
        "right_stride_segments": right_stride_segments,
    }


def compute_ankle_relative_xy(keypoints_all, origin_foot="left", normalize_y=False):
    """
    Compute relative ankle trajectory with one foot as origin.

    Returns:
        x_diff: moving_foot_x - origin_foot_x
        y_diff: moving_foot_y - origin_foot_y   (raw or normalized)
    """
    left_ankle = keypoints_all[:, JOINT_IDX["left_ankle"], :]   # (T,2)
    right_ankle = keypoints_all[:, JOINT_IDX["right_ankle"], :] # (T,2)

    if origin_foot == "left":
        origin_xy = left_ankle
        moving_xy = right_ankle
        moving_name = "right"
    elif origin_foot == "right":
        origin_xy = right_ankle
        moving_xy = left_ankle
        moving_name = "left"
    else:
        raise ValueError("origin_foot must be 'left' or 'right'")

    x_diff = moving_xy[:, 0] - origin_xy[:, 0]

    if not normalize_y:
        y_diff = moving_xy[:, 1] - origin_xy[:, 1]
    else:
        # normalize each ankle y by torso scale first, then subtract
        if moving_name == "right":
            moving_y_norm = compute_normalized_joint_y_series(keypoints_all, "right_ankle")
            origin_y_norm = compute_normalized_joint_y_series(keypoints_all, "left_ankle")
        else:
            moving_y_norm = compute_normalized_joint_y_series(keypoints_all, "left_ankle")
            origin_y_norm = compute_normalized_joint_y_series(keypoints_all, "right_ankle")

        y_diff = moving_y_norm - origin_y_norm

    return x_diff, y_diff


def plot_stride_loops_xydiff(frames,
                             keypoints_all,
                             stride_event_idx,
                             origin_foot="left",
                             normalize_y=False,
                             title=None,
                             save_path=None):
    """
    Plot Y_diff vs X_diff, one loop per stride.
    stride_event_idx should be boundary events of the SAME foot,
    e.g. consecutive left minima define one stride.
    """
    x_diff, y_diff = compute_ankle_relative_xy(
        keypoints_all,
        origin_foot=origin_foot,
        normalize_y=normalize_y
    )

    plt.figure(figsize=(8, 8))

    n_stride = 0
    for i in range(len(stride_event_idx) - 1):
        s = stride_event_idx[i]
        e = stride_event_idx[i + 1]

        if e <= s:
            continue

        plt.plot(
            x_diff[s:e+1],
            y_diff[s:e+1],
            linewidth=2,
            label=f"stride {i}"
        )

        # mark start point
        plt.scatter(
            x_diff[s],
            y_diff[s],
            marker="o",
            s=35
        )

        n_stride += 1

    plt.axhline(0, color="gray", linewidth=1, alpha=0.4)
    plt.axvline(0, color="gray", linewidth=1, alpha=0.4)

    plt.xlabel("X_diff (moving foot - origin foot)")
    plt.ylabel("Y_diff" + (" (Y normalized)" if normalize_y else " (raw)"))

    if title is None:
        if normalize_y:
            title = f"Stride Loops: Y_diff vs X_diff ({origin_foot} foot as origin, Y normalized)"
        else:
            title = f"Stride Loops: Y_diff vs X_diff ({origin_foot} foot as origin, raw)"
    plt.title(title)

    plt.grid(True, alpha=0.3)
    if n_stride > 0 and n_stride <= 15:
        plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


def plot_stride_loops_raw_and_norm(frames,
                                   keypoints_all,
                                   stride_event_idx,
                                   origin_foot="left",
                                   save_path=None):
    """
    Make a 1x2 figure:
      left: raw Y_diff vs X_diff
      right: normalized-Y Y_diff vs X_diff
    """
    x_diff_raw, y_diff_raw = compute_ankle_relative_xy(
        keypoints_all,
        origin_foot=origin_foot,
        normalize_y=False
    )
    x_diff_norm, y_diff_norm = compute_ankle_relative_xy(
        keypoints_all,
        origin_foot=origin_foot,
        normalize_y=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # raw
    for i in range(len(stride_event_idx) - 1):
        s = stride_event_idx[i]
        e = stride_event_idx[i + 1]
        if e <= s:
            continue
        axes[0].plot(x_diff_raw[s:e+1], y_diff_raw[s:e+1], linewidth=2, label=f"stride {i}")
        axes[0].scatter(x_diff_raw[s], y_diff_raw[s], s=25)

    axes[0].axhline(0, color="gray", linewidth=1, alpha=0.4)
    axes[0].axvline(0, color="gray", linewidth=1, alpha=0.4)
    axes[0].set_xlabel("X_diff")
    axes[0].set_ylabel("Y_diff (raw)")
    axes[0].set_title(f"Raw: {origin_foot} foot as origin")
    axes[0].grid(True, alpha=0.3)

    # normalized Y
    for i in range(len(stride_event_idx) - 1):
        s = stride_event_idx[i]
        e = stride_event_idx[i + 1]
        if e <= s:
            continue
        axes[1].plot(x_diff_norm[s:e+1], y_diff_norm[s:e+1], linewidth=2, label=f"stride {i}")
        axes[1].scatter(x_diff_norm[s], y_diff_norm[s], s=25)

    axes[1].axhline(0, color="gray", linewidth=1, alpha=0.4)
    axes[1].axvline(0, color="gray", linewidth=1, alpha=0.4)
    axes[1].set_xlabel("X_diff")
    axes[1].set_ylabel("Y_diff (Y normalized)")
    axes[1].set_title(f"Y-normalized: {origin_foot} foot as origin")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Stride Loops: Y_diff vs X_diff")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()

def analyze_stride_loops(frames,
                         keypoints_all,
                         stride_event_idx,
                         origin_foot="left",
                         save_prefix=None):
    """
    Plot:
      1) raw Y_diff vs X_diff
      2) normalized-Y Y_diff vs X_diff
      3) combined 1x2 figure
    """
    plot_stride_loops_xydiff(
        frames,
        keypoints_all,
        stride_event_idx,
        origin_foot=origin_foot,
        normalize_y=False,
        save_path=None if save_prefix is None else f"{save_prefix}_loop_raw.png"
    )

    plot_stride_loops_xydiff(
        frames,
        keypoints_all,
        stride_event_idx,
        origin_foot=origin_foot,
        normalize_y=True,
        save_path=None if save_prefix is None else f"{save_prefix}_loop_y_norm.png"
    )

    plot_stride_loops_raw_and_norm(
        frames,
        keypoints_all,
        stride_event_idx,
        origin_foot=origin_foot,
        save_path=None if save_prefix is None else f"{save_prefix}_loop_raw_vs_norm.png"
    )


def detect_peaks_from_feet_y_diff(frames,
                                  keypoints_all,
                                  fps=30.0,
                                  use_normalized_y=False,
                                  bias_correct=False,
                                  smooth_window=15,
                                  smooth_polyorder=2,
                                  min_distance_sec=0.5,
                                  prominence=0.03):
    """
    For front/back videos:
    y_diff(t) = left_ankle_y - right_ankle_y
    detect positive and negative peaks.
    """
    if use_normalized_y:
        left_y = compute_normalized_joint_y_series(keypoints_all, "left_ankle")
        right_y = compute_normalized_joint_y_series(keypoints_all, "right_ankle")
    else:
        left_y = keypoints_all[:, JOINT_IDX["left_ankle"], 1]
        right_y = keypoints_all[:, JOINT_IDX["right_ankle"], 1]

    y_diff = left_y - right_y
    y_diff_bias = float(y_diff[0]) if bias_correct and len(y_diff) > 0 else 0.0
    y_diff = y_diff - y_diff_bias

    y_diff_smooth = smooth_signal(
        y_diff,
        window_length=smooth_window,
        polyorder=smooth_polyorder
    )

    min_distance_frames = max(3, int(min_distance_sec * fps))

    pos_peaks, pos_props = find_peaks(
        y_diff_smooth,
        distance=min_distance_frames,
        prominence=prominence
    )
    neg_peaks, neg_props = find_peaks(
        -y_diff_smooth,
        distance=min_distance_frames,
        prominence=prominence
    )

    return {
        "y_diff_raw": y_diff,
        "y_diff_smooth": y_diff_smooth,
        "y_diff_bias": y_diff_bias,
        "bias_correct": bias_correct,
        "pos_peaks": pos_peaks,
        "neg_peaks": neg_peaks,
        "pos_props": pos_props,
        "neg_props": neg_props,
    }


def plot_feet_y_diff_with_peaks(frames,
                                y_diff_raw,
                                y_diff_smooth,
                                pos_peaks,
                                neg_peaks,
                                use_normalized_y=False,
                                save_path=None):
    plt.figure(figsize=(14, 5))

    plt.plot(frames, y_diff_raw, alpha=0.3, label="left-right ankle Y diff raw")
    plt.plot(frames, y_diff_smooth, linewidth=2, label="left-right ankle Y diff smooth")

    if len(pos_peaks) > 0:
        plt.scatter(
            frames[pos_peaks],
            y_diff_smooth[pos_peaks],
            marker="^",
            s=80,
            label="positive peaks"
        )

    if len(neg_peaks) > 0:
        plt.scatter(
            frames[neg_peaks],
            y_diff_smooth[neg_peaks],
            marker="v",
            s=80,
            label="negative peaks"
        )

    plt.axhline(0, color="gray", linewidth=1, alpha=0.4)
    plt.xlabel("Time (frame)")
    plt.ylabel("Left ankle Y - Right ankle Y" + (" (normalized)" if use_normalized_y else ""))
    plt.title("Feet Y Difference vs Time with Peak Detection")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


def plot_feet_y_diff_with_peaks_and_gt(frames,
                                       y_diff_raw,
                                       y_diff_smooth,
                                       pos_peaks,
                                       neg_peaks,
                                       gt_json_path,
                                       use_normalized_y=False,
                                       pos_gt_label="left GT",
                                       neg_gt_label="right GT",
                                       save_path=None):
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    frame_min, frame_max = frames[0], frames[-1]

    gt_pos = [f for f in gt.get("left_events", []) if frame_min <= f <= frame_max]
    gt_neg = [f for f in gt.get("right_events", []) if frame_min <= f <= frame_max]

    plt.figure(figsize=(16, 6))

    plt.plot(frames, y_diff_raw, alpha=0.28, label="left-right ankle Y diff raw")
    plt.plot(frames, y_diff_smooth, linewidth=2, label="left-right ankle Y diff smooth")

    if len(pos_peaks) > 0:
        plt.scatter(
            frames[pos_peaks],
            y_diff_smooth[pos_peaks],
            marker="^",
            s=90,
            label="positive peaks"
        )

    if len(neg_peaks) > 0:
        plt.scatter(
            frames[neg_peaks],
            y_diff_smooth[neg_peaks],
            marker="v",
            s=90,
            label="negative peaks"
        )

    for i, f in enumerate(gt_pos):
        plt.axvline(
            x=f,
            color="blue",
            linestyle="--",
            alpha=0.35,
            label=pos_gt_label if i == 0 else None
        )

    for i, f in enumerate(gt_neg):
        plt.axvline(
            x=f,
            color="red",
            linestyle="--",
            alpha=0.35,
            label=neg_gt_label if i == 0 else None
        )

    plt.axhline(0, color="gray", linewidth=1, alpha=0.4)
    plt.xlabel("Time (frame)")
    plt.ylabel("Left ankle Y - Right ankle Y" + (" (normalized)" if use_normalized_y else ""))
    plt.title("Feet Y Difference vs Time with Positive/Negative Peaks and GT")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()

def analyze_front_back_stride_by_y_diff_with_gt(frames,
                                                keypoints_all,
                                                gt_json_path,
                                                fps=30.0,
                                                use_normalized_y=False,
                                                bias_correct=False,
                                                save_prefix=None):
    results = detect_peaks_from_feet_y_diff(
        frames,
        keypoints_all,
        fps=fps,
        use_normalized_y=use_normalized_y,
        bias_correct=bias_correct,
        smooth_window=15,
        smooth_polyorder=2,
        min_distance_sec=0.5,
        prominence=0.03 if use_normalized_y else 3.0
    )

    plot_feet_y_diff_with_peaks_and_gt(
        frames,
        results["y_diff_raw"],
        results["y_diff_smooth"],
        results["pos_peaks"],
        results["neg_peaks"],
        gt_json_path=gt_json_path,
        use_normalized_y=use_normalized_y,
        pos_gt_label="left GT",
        neg_gt_label="right GT",
        save_path=None if save_prefix is None else (
            f"{save_prefix}_feet_y_diff_with_gt_norm.png"
            if use_normalized_y else
            f"{save_prefix}_feet_y_diff_with_gt_raw.png"
        )
    )

    print("\nFront/back stride candidates from feet Y difference:")
    print(f"  bias correction: {bias_correct}, bias = {results['y_diff_bias']:.6f}")
    print(f"  positive peaks: {frames[results['pos_peaks']] if len(results['pos_peaks']) > 0 else []}")
    print(f"  negative peaks: {frames[results['neg_peaks']] if len(results['neg_peaks']) > 0 else []}")

    step_metrics = compute_zero_crossing_step_metrics(
        frames,
        results["y_diff_smooth"],
        results["pos_peaks"],
        results["neg_peaks"],
        fps=fps,
    )

    left_step_times_frames = step_metrics["left"]["step_times_frames"]
    right_step_times_frames = step_metrics["right"]["step_times_frames"]
    left_step_times = left_step_times_frames / fps
    right_step_times = right_step_times_frames / fps
    left_step_heights = step_metrics["left"]["step_heights"]
    right_step_heights = step_metrics["right"]["step_heights"]
    left_step_areas = step_metrics["left"]["step_areas_seconds"]
    right_step_areas = step_metrics["right"]["step_areas_seconds"]
    curve_integrals = compute_curve_integrals(
        frames,
        results["y_diff_smooth"],
        fps=fps
    )

    print("\nZero-crossing step metrics:")
    print(f"  raw zero crossings: {step_metrics['raw_zero_frames']}")
    print(f"  merged zero crossings: {step_metrics['zero_frames']}")
    print(f"  effective zero crossings: {step_metrics['effective_zero_frames']}")
    print(f"  valid left peaks: {frames[step_metrics['left']['valid_peaks']] if len(step_metrics['left']['valid_peaks']) > 0 else []}")
    print(f"  skipped left peaks: {frames[step_metrics['left']['skipped_peaks']] if len(step_metrics['left']['skipped_peaks']) > 0 else []}")
    print(f"  valid right peaks: {frames[step_metrics['right']['valid_peaks']] if len(step_metrics['right']['valid_peaks']) > 0 else []}")
    print(f"  skipped right peaks: {frames[step_metrics['right']['skipped_peaks']] if len(step_metrics['right']['skipped_peaks']) > 0 else []}")

    print("\nY-diff curve integrals:")
    print(f"  signed integral: {curve_integrals['signed_area_seconds']:.6f}")
    print(f"  absolute integral: {curve_integrals['absolute_area_seconds']:.6f}")
    print(f"  left step integrals: {left_step_areas}")
    print(f"  right step integrals: {right_step_areas}")
    summarize_metric(left_step_areas, "Left step integral")
    summarize_metric(right_step_areas, "Right step integral")
    print(f"  left total integral = {float(np.nansum(left_step_areas)):.6f}")
    print(f"  right total integral = {float(np.nansum(right_step_areas)):.6f}")

    plot_left_right_step_metrics(
        left_step_times,
        right_step_times,
        left_step_heights,
        right_step_heights,
        save_prefix=save_prefix
    )

    plot_left_right_step_scatter(
        left_step_times,
        right_step_times,
        left_step_heights,
        right_step_heights,
        save_path=None if save_prefix is None else f"{save_prefix}_step_scatter.png"
    )

    compare_left_right_step_metrics(
        left_step_times,
        right_step_times,
        left_step_heights,
        right_step_heights
    )

    results.update({
        "raw_zero_indices": step_metrics["raw_zero_indices"],
        "raw_zero_frames": step_metrics["raw_zero_frames"],
        "zero_indices": step_metrics["zero_indices"],
        "zero_frames": step_metrics["zero_frames"],
        "effective_zero_indices": step_metrics["effective_zero_indices"],
        "effective_zero_frames": step_metrics["effective_zero_frames"],
        "left_step_times": left_step_times,
        "right_step_times": right_step_times,
        "left_step_times_frames": left_step_times_frames,
        "right_step_times_frames": right_step_times_frames,
        "left_step_heights": left_step_heights,
        "right_step_heights": right_step_heights,
        "curve_signed_integral": curve_integrals["signed_area_seconds"],
        "curve_absolute_integral": curve_integrals["absolute_area_seconds"],
        "curve_signed_integral_frames": curve_integrals["signed_area_frames"],
        "curve_absolute_integral_frames": curve_integrals["absolute_area_frames"],
        "left_step_integrals": left_step_areas,
        "right_step_integrals": right_step_areas,
        "left_step_integrals_frames": step_metrics["left"]["step_areas_frames"],
        "right_step_integrals_frames": step_metrics["right"]["step_areas_frames"],
        "left_step_integral_mean": float(np.nanmean(left_step_areas)) if len(left_step_areas) > 0 else np.nan,
        "left_step_integral_std": float(np.nanstd(left_step_areas)) if len(left_step_areas) > 0 else np.nan,
        "right_step_integral_mean": float(np.nanmean(right_step_areas)) if len(right_step_areas) > 0 else np.nan,
        "right_step_integral_std": float(np.nanstd(right_step_areas)) if len(right_step_areas) > 0 else np.nan,
        "left_step_integral_total": float(np.nansum(left_step_areas)),
        "right_step_integral_total": float(np.nansum(right_step_areas)),
        "left_step_segments": step_metrics["left"]["step_segments"],
        "right_step_segments": step_metrics["right"]["step_segments"],
        "left_valid_peak_idx": step_metrics["left"]["valid_peaks"],
        "right_valid_peak_idx": step_metrics["right"]["valid_peaks"],
        "left_skipped_peak_idx": step_metrics["left"]["skipped_peaks"],
        "right_skipped_peak_idx": step_metrics["right"]["skipped_peaks"],
        "positive_peak_frames": frames[results["pos_peaks"]],
        "negative_peak_frames": frames[results["neg_peaks"]],
        "left_valid_peak_frames": frames[step_metrics["left"]["valid_peaks"]],
        "right_valid_peak_frames": frames[step_metrics["right"]["valid_peaks"]],
        "left_skipped_peak_frames": frames[step_metrics["left"]["skipped_peaks"]],
        "right_skipped_peak_frames": frames[step_metrics["right"]["skipped_peaks"]],
    })

    return results

def analyze_front_back_stride_by_y_diff(frames,
                                        keypoints_all,
                                        fps=30.0,
                                        use_normalized_y=False,
                                        bias_correct=False,
                                        save_prefix=None):
    results = detect_peaks_from_feet_y_diff(
        frames,
        keypoints_all,
        fps=fps,
        use_normalized_y=use_normalized_y,
        bias_correct=bias_correct,
        smooth_window=15,
        smooth_polyorder=2,
        min_distance_sec=0.25,
        prominence=0.03 if use_normalized_y else 3.0
    )

    plot_feet_y_diff_with_peaks(
        frames,
        results["y_diff_raw"],
        results["y_diff_smooth"],
        results["pos_peaks"],
        results["neg_peaks"],
        use_normalized_y=use_normalized_y,
        save_path=None if save_prefix is None else (
            f"{save_prefix}_feet_y_diff_peaks_norm.png"
            if use_normalized_y else
            f"{save_prefix}_feet_y_diff_peaks_raw.png"
        )
    )

    print("\nFront/back stride candidates from feet Y difference:")
    print(f"  bias correction: {bias_correct}, bias = {results['y_diff_bias']:.6f}")
    print(f"  positive peaks: {frames[results['pos_peaks']] if len(results['pos_peaks']) > 0 else []}")
    print(f"  negative peaks: {frames[results['neg_peaks']] if len(results['neg_peaks']) > 0 else []}")

    curve_integrals = compute_curve_integrals(
        frames,
        results["y_diff_smooth"],
        fps=fps
    )
    print("\nY-diff curve integrals:")
    print(f"  signed integral: {curve_integrals['signed_area_seconds']:.6f}")
    print(f"  absolute integral: {curve_integrals['absolute_area_seconds']:.6f}")

    results.update({
        "curve_signed_integral": curve_integrals["signed_area_seconds"],
        "curve_absolute_integral": curve_integrals["absolute_area_seconds"],
        "curve_signed_integral_frames": curve_integrals["signed_area_frames"],
        "curve_absolute_integral_frames": curve_integrals["absolute_area_frames"],
    })

    return results

def plot_left_right_ankle_waves_with_gt(
    frames,
    left_raw, left_smooth, left_pred_idx,
    right_raw, right_smooth, right_pred_idx,
    gt_json_path,
    save_path=None
):
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    frame_min, frame_max = frames[0], frames[-1]

    gt_left = [f for f in gt.get("left_events", []) if frame_min <= f <= frame_max]
    gt_right = [f for f in gt.get("right_events", []) if frame_min <= f <= frame_max]

    plt.figure(figsize=(16, 7))

    plt.plot(frames, left_raw, label="left ankle raw", alpha=0.25, color="skyblue")
    plt.plot(frames, left_smooth, label="left ankle smooth", linewidth=2, color="blue")

    plt.plot(frames, right_raw, label="right ankle raw", alpha=0.25, color="lightcoral")
    plt.plot(frames, right_smooth, label="right ankle smooth", linewidth=2, color="red")

    if len(left_pred_idx) > 0:
        plt.scatter(frames[left_pred_idx], left_smooth[left_pred_idx],
                    marker="x", s=80, color="blue", label="left prediction")

    if len(right_pred_idx) > 0:
        plt.scatter(frames[right_pred_idx], right_smooth[right_pred_idx],
                    marker="x", s=80, color="red", label="right prediction")

    for i, f in enumerate(gt_left):
        plt.axvline(x=f, color="blue", linestyle="--", alpha=0.35,
                    label="left GT" if i == 0 else None)

    for i, f in enumerate(gt_right):
        plt.axvline(x=f, color="red", linestyle="--", alpha=0.35,
                    label="right GT" if i == 0 else None)

    plt.xlabel("Time (frame)")
    plt.ylabel("Normalized Y")
    plt.title("Left/Right Ankle Waves with GT and Prediction")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()

def compute_relative_loop_phase(keypoints_all, origin_foot="left", normalize_y=True, smooth_xy=True):
    """
    Build relative ankle loop phase from x_diff and y_diff.
    """
    x_diff, y_diff = compute_ankle_relative_xy(
        keypoints_all,
        origin_foot=origin_foot,
        normalize_y=normalize_y
    )

    x_diff = interpolate_nan(x_diff)
    y_diff = interpolate_nan(y_diff)

    if smooth_xy:
        x_diff = smooth_signal(x_diff, window_length=11, polyorder=2)
        y_diff = smooth_signal(y_diff, window_length=11, polyorder=2)

    # center the loop
    x_centered = x_diff - np.median(x_diff)
    y_centered = y_diff - np.median(y_diff)

    # normalize scale so that phase is more stable
    x_scale = np.std(x_centered)
    y_scale = np.std(y_centered)

    if x_scale < 1e-8:
        x_scale = 1.0
    if y_scale < 1e-8:
        y_scale = 1.0

    x_centered = x_centered / x_scale
    y_centered = y_centered / y_scale

    theta_wrapped = np.arctan2(y_centered, x_centered)
    theta_unwrapped = np.unwrap(theta_wrapped)

    return x_diff, y_diff, x_centered, y_centered, theta_wrapped, theta_unwrapped

def detect_stride_events_from_loop_phase(frames,
                                         keypoints_all,
                                         origin_foot="left",
                                         normalize_y=True,
                                         fps=30.0,
                                         min_stride_sec=0.4,
                                         phase_offset=0.0):
    """
    Detect stride events directly from the relative ankle loop phase.
    Each time the unwrapped phase crosses phase_offset + 2*pi*k,
    we treat it as one stride boundary.
    """
    x_diff, y_diff, x_centered, y_centered, theta_wrapped, theta_unwrapped = compute_relative_loop_phase(
        keypoints_all,
        origin_foot=origin_foot,
        normalize_y=normalize_y,
        smooth_xy=True
    )

    min_distance_frames = max(3, int(min_stride_sec * fps))

    shifted_phase = theta_unwrapped - phase_offset
    cycle_id = np.floor(shifted_phase / (2 * np.pi)).astype(int)

    event_idx = []
    last_idx = -10**9

    for i in range(1, len(cycle_id)):
        if cycle_id[i] > cycle_id[i - 1]:
            if i - last_idx >= min_distance_frames:
                event_idx.append(i)
                last_idx = i

    event_idx = np.array(event_idx, dtype=np.int32)

    return {
        "x_diff": x_diff,
        "y_diff": y_diff,
        "x_centered": x_centered,
        "y_centered": y_centered,
        "theta_wrapped": theta_wrapped,
        "theta_unwrapped": theta_unwrapped,
        "event_idx": event_idx,
    }


def plot_loop_phase_and_events(frames,
                               theta_wrapped,
                               theta_unwrapped,
                               event_idx,
                               save_path=None):
    plt.figure(figsize=(14, 6))

    plt.plot(frames, theta_wrapped, label="wrapped phase", alpha=0.45)
    plt.plot(frames, theta_unwrapped, label="unwrapped phase", linewidth=2)

    if len(event_idx) > 0:
        plt.scatter(
            frames[event_idx],
            theta_unwrapped[event_idx],
            marker="x",
            s=80,
            label="detected stride events"
        )

    plt.xlabel("Frame")
    plt.ylabel("Phase (radian)")
    plt.title("Relative Foot Loop Phase with Detected Stride Events")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()

def plot_loop_with_detected_events(keypoints_all,
                                   event_idx,
                                   origin_foot="left",
                                   normalize_y=True,
                                   save_path=None):
    x_diff, y_diff = compute_ankle_relative_xy(
        keypoints_all,
        origin_foot=origin_foot,
        normalize_y=normalize_y
    )

    plt.figure(figsize=(8, 8))
    plt.plot(x_diff, y_diff, alpha=0.4, linewidth=1.5, label="full loop trajectory")

    if len(event_idx) > 0:
        plt.scatter(
            x_diff[event_idx],
            y_diff[event_idx],
            s=80,
            marker="x",
            label="detected events"
        )

    plt.axhline(0, color="gray", linewidth=1, alpha=0.4)
    plt.axvline(0, color="gray", linewidth=1, alpha=0.4)
    plt.xlabel("X_diff")
    plt.ylabel("Y_diff" + (" (Y normalized)" if normalize_y else " (raw)"))
    plt.title(f"Loop Trajectory with Detected Events ({origin_foot} foot as origin)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()

def analyze_stride_events_from_loops(frames,
                                     keypoints_all,
                                     fps=30.0,
                                     origin_foot="left",
                                     normalize_y=True,
                                     min_stride_sec=0.4,
                                     phase_offset=0.0,
                                     save_prefix=None):
    results = detect_stride_events_from_loop_phase(
        frames,
        keypoints_all,
        origin_foot=origin_foot,
        normalize_y=normalize_y,
        fps=fps,
        min_stride_sec=min_stride_sec,
        phase_offset=phase_offset
    )

    plot_loop_phase_and_events(
        frames,
        results["theta_wrapped"],
        results["theta_unwrapped"],
        results["event_idx"],
        save_path=None if save_prefix is None else f"{save_prefix}_loop_phase.png"
    )

    plot_loop_with_detected_events(
        keypoints_all,
        results["event_idx"],
        origin_foot=origin_foot,
        normalize_y=normalize_y,
        save_path=None if save_prefix is None else f"{save_prefix}_loop_events_on_xy.png"
    )

    # optional: now use loop-based events to segment and draw one loop per stride
    if len(results["event_idx"]) >= 2:
        plot_stride_loops_xydiff(
            frames,
            keypoints_all,
            stride_event_idx=results["event_idx"],
            origin_foot=origin_foot,
            normalize_y=normalize_y,
            save_path=None if save_prefix is None else f"{save_prefix}_segmented_loops.png"
        )

    print("\nDetected stride events from loop phase:")
    print(frames[results["event_idx"]] if len(results["event_idx"]) > 0 else [])

    return results

def save_loop_motion_video(frames,
                           keypoints_all,
                           origin_foot="left",
                           normalize_y=True,
                           fps=30.0,
                           tail_length=None,
                           bias_correct=False,
                           beta_x_normalize=False,
                           min_stride_sec=0.4,
                           save_path="loop_motion.mp4",
                           save_prefix=None):
    """
    Save a video showing how the Y_diff vs X_diff trajectory evolves over time.

    Parameters
    ----------
    frames : array of frame ids
    keypoints_all : (T,17,2)
    origin_foot : "left" or "right"
    normalize_y : whether to normalize Y before computing Y_diff
    fps : output video fps
    tail_length : if not None, only show the recent N points as a trailing tail
    bias_correct : if True, shift the first point to (0,0)
    beta_x_normalize : if True, apply beta_i = (1-alpha_i)*(x0/xn)+alpha_i to x_diff
    min_stride_sec : minimum time between winding-based stride events
    save_path : output mp4 path
    """
    if save_prefix is not None:
        save_path = f"{save_prefix}.mp4"

    x_diff, y_diff = compute_ankle_relative_xy(
        keypoints_all,
        origin_foot=origin_foot,
        normalize_y=normalize_y
    )

    x_diff = interpolate_nan(x_diff)
    y_diff = interpolate_nan(y_diff)

    # optional smoothing so the motion is easier to watch
    x_plot = smooth_signal(x_diff, window_length=11, polyorder=2)
    y_plot = smooth_signal(y_diff, window_length=11, polyorder=2)

    if beta_x_normalize and len(x_plot) > 1:
        n = len(x_plot) - 1
        i_arr = np.arange(len(x_plot), dtype=np.float32)
        alpha = 1.0 - (i_arr / float(n))
        x0 = float(x_plot[0])
        xn = float(x_plot[-1])
        if abs(xn) >= 1e-8:
            beta = (1.0 - alpha) * (x0 / xn) + alpha
            x_plot = x_plot * beta

    if bias_correct and len(x_plot) > 0:
        x_plot = x_plot - x_plot[0]
        y_plot = y_plot - y_plot[0]

    radius = np.sqrt(x_plot * x_plot + y_plot * y_plot)
    max_radius = float(np.nanmax(radius)) if len(radius) > 0 else 0.0
    valid_radius = radius > max(1e-6, 0.02 * max_radius)

    theta_wrapped = np.arctan2(y_plot, x_plot)
    theta_unwrapped = np.unwrap(theta_wrapped)

    if np.any(valid_radius):
        first_valid = int(np.where(valid_radius)[0][0])
    else:
        first_valid = 0

    theta_unwrapped = theta_unwrapped - theta_unwrapped[first_valid]
    total_delta = float(theta_unwrapped[-1] - theta_unwrapped[first_valid]) if len(theta_unwrapped) > 1 else 0.0
    direction = 1.0 if total_delta >= 0 else -1.0
    winding_phase = direction * theta_unwrapped
    winding_phase = winding_phase - winding_phase[first_valid]
    winding_factor = winding_phase / (2.0 * np.pi)

    min_distance_frames = max(3, int(min_stride_sec * fps))
    cycle_id = np.floor(winding_factor).astype(int)
    event_idx = []
    last_event_idx = -10**9
    for idx in range(max(1, first_valid + 1), len(cycle_id)):
        if cycle_id[idx] > cycle_id[idx - 1] and idx - last_event_idx >= min_distance_frames:
            event_idx.append(idx)
            last_event_idx = idx
    event_idx = np.array(event_idx, dtype=np.int32)

    fig, ax = plt.subplots(figsize=(7, 7))

    # set fixed axis range
    x_margin = 0.1 * max(1e-6, np.max(x_plot) - np.min(x_plot))
    y_margin = 0.1 * max(1e-6, np.max(y_plot) - np.min(y_plot))

    ax.set_xlim(np.min(x_plot) - x_margin, np.max(x_plot) + x_margin)
    ax.set_ylim(np.min(y_plot) - y_margin, np.max(y_plot) + y_margin)

    ax.axhline(0, color="gray", linewidth=1, alpha=0.4)
    ax.axvline(0, color="gray", linewidth=1, alpha=0.4)

    ax.set_xlabel("X_diff (moving foot - origin foot)" + (" shifted" if bias_correct else ""))
    ax.set_ylabel("Y_diff" + (" (Y normalized)" if normalize_y else " (raw)") + (" shifted" if bias_correct else ""))
    ax.set_title(f"Loop Motion Over Time ({origin_foot} foot as origin)")
    ax.grid(True, alpha=0.3)

    line, = ax.plot([], [], linewidth=2, label="trajectory")
    point, = ax.plot([], [], marker="o", markersize=8, linestyle="None", label="current point")
    event_points, = ax.plot([], [], marker="x", markersize=9, linestyle="None", label="2pi winding events")
    text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    ax.legend()

    def init():
        line.set_data([], [])
        point.set_data([], [])
        event_points.set_data([], [])
        text.set_text("")
        return line, point, event_points, text

    def update(i):
        if tail_length is None:
            s = 0
        else:
            s = max(0, i - tail_length + 1)

        line.set_data(x_plot[s:i+1], y_plot[s:i+1])
        point.set_data([x_plot[i]], [y_plot[i]])
        visible_events = event_idx[event_idx <= i]
        if len(visible_events) > 0:
            event_points.set_data(x_plot[visible_events], y_plot[visible_events])
        else:
            event_points.set_data([], [])

        current_winding = winding_factor[i] if len(winding_factor) > i else np.nan
        text.set_text(
            f"frame = {frames[i]}\n"
            f"winding = {current_winding:.2f}\n"
            f"2pi events = {len(visible_events)}"
        )
        return line, point, event_points, text

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        init_func=init,
        interval=1000.0 / fps,
        blit=True
    )

    writer = FFMpegWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=160)
    plt.close(fig)

    print(f"Saved loop motion video to: {save_path}")
    print(f"Winding 2pi event frames in video: {frames[event_idx] if len(event_idx) > 0 else []}")



if __name__ == "__main__":
    # metrics_dir = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Chiocchi_mirror\metrics"
    # os.makedirs(metrics_dir, exist_ok=True)
    # track_json_path = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Chiocchi_mirror\track.json"
    # save_path = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Chiocchi_mirror\17_keypoints_time_vs_y.png"
    # fps = 30.0
    # start_frame = 30
    # end_frame = 480

    metrics_dir = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Baseline_side\metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    track_json_path = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Baseline_side\tracked\track.json"
    save_path = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Baseline_side\metrics\17_keypoints_time_vs_y.png"
    fps = 30.0
    start_frame = 20
    end_frame = 115

    # metrics_dir = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Sronce_walker_post_op\metrics"
    # os.makedirs(metrics_dir, exist_ok=True)
    # track_json_path = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Sronce_walker_post_op\tracked\track.json"
    # save_path = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Sronce_walker_post_op\metrics\17_keypoints_time_vs_y.png"
    # fps = 60.0
    # start_frame = None
    # end_frame = 300

    # metrics_dir = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Sronce_preop\metrics"
    # os.makedirs(metrics_dir, exist_ok=True)
    # track_json_path = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Sronce_preop\tracked\track.json"
    # save_path = os.path.join(metrics_dir, "17_keypoints_time_vs_y.png")
    # fps = 60.0
    # start_frame = None
    # end_frame = None

    # metrics_dir = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Baseline\metrics"
    # os.makedirs(metrics_dir, exist_ok=True)
    # track_json_path = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Baseline\tracked\track.json"
    # save_path = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Baseline\metrics\17_keypoints_time_vs_y.png"
    # fps = 24.0
    # start_frame = None
    # end_frame = None

    # metrics_dir = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Ovcharenko_preop\metrics"
    # os.makedirs(metrics_dir, exist_ok=True)
    # track_json_path = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Ovcharenko_preop\tracked\track.json"
    # save_path = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Ovcharenko_preop\metrics\17_keypoints_time_vs_y.png"
    # fps = 60.0
    # start_frame = 400
    # end_frame = None

    frames, keypoints_all = load_track_json(track_json_path)


    frames, keypoints_all = crop_frame_range(
        frames,
        keypoints_all,
        start_frame=start_frame,
        end_frame=end_frame
    )

    print(f"Using frames from {frames[0]} to {frames[-1]} (total {len(frames)} frames)")

    # Step 1: plot all joints time vs y and print top joints by range and std
    stats = plot_all_joints_time_vs_y(frames, keypoints_all, save_path)
    print_top_joints(stats, top_k=5, metric="range")
    print_top_joints(stats, top_k=5, metric="std")

    # # Step 2: analyze left/right ankle separately
    # gait_results = analyze_ankle_stride(
    #     frames,
    #     keypoints_all,
    #     fps=fps,
    #     save_prefix=os.path.join(metrics_dir, "gait")
    # )

    # Step 2.5: save loop motion videos (optional, only for videos shooting from the side)
    save_loop_motion_video(
        frames,
        keypoints_all,
        origin_foot="left",
        normalize_y=True,
        fps=fps,
        tail_length=None,  # show recent 60 points only; set None to show full history
        bias_correct=True,
        beta_x_normalize=True,
        save_prefix=r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Baseline_side\metrics\loop_motion"
    )

    # Step 3: front/back stride detection from feet Y difference (normalized)
    #  with GT
    fb_norm = analyze_front_back_stride_by_y_diff_with_gt(
        frames,
        keypoints_all,
        gt_json_path=r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Baseline_side\tracked\a_manual_stride_events.json",
        fps=fps,
        use_normalized_y=True,
        bias_correct=False,
        save_prefix=os.path.join(metrics_dir, "frontback_norm")
    )

    # Step 4: wrist relative height
    wrist_results = analyze_wrist_relative_height(
        frames,
        keypoints_all,
        save_path=os.path.join(metrics_dir, "wrist_relative_height.png")
    )

    write_metrics_report(
        os.path.join(metrics_dir, "metrics.txt"),
        fb_norm,
        wrist_results=wrist_results
    )
    
    # # Step 5: plot ankle waves with GT stride events
    # plot_left_right_ankle_waves_with_gt(
    #     frames,
    #     gait_results["left_ankle_norm"],
    #     gait_results["left_ankle_smooth"],
    #     gait_results["left_minima_idx"],
    #     gait_results["right_ankle_norm"],
    #     gait_results["right_ankle_smooth"],
    #     gait_results["right_minima_idx"],
    #     gt_json_path=r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Sronce_preop\tracked\a_manual_stride_events.json",
    #     save_path=os.path.join(metrics_dir, "strides_with_gt.png")
    # )

# Knee distance : abs(x_LKnee - x_RKnee) or euclidean(LKnee, RKnee) & draw knee_distance vs frame  Use left&right ankle
# left/right swing phase symmetry: smoothing/detrending the curves: patient is getting close/away from camera & camera may move for TestData
# Hand pos wrt shoulder: dist(wrist, shoulder)

# Find stride variability (time & wave height) draw a graph height vs stride time. For regular people points should be dense, for patients, it should be more sparse.
# Find std of stride time and stride height/wave height
# We can invert the curve and find the peaks using scipy.signal.find_peaks to find the distance between local minima (stride time)
# Wrist relative y distance to shoulder and hip with linear interpolation with shoulder as 1 and hip as 0
# Use the same interpolation to find the coordinate of knees and ankles so as to minimize the impact of closer/further away from the camera.


# Ground truth for 2-3 videos Visualize GT (Draw Vertical lines) and our prediction in the same graph Time vs Y Coordinates

# Plot Y_diff vs X_diff on the same graph with one foot as the origin or a fixing point to see if there's some kind of a circular pattern with no normalization vs with Y coordinate normalization and each circle should represent one stride
# For those videos shot in the back/front, consider the Y_diff of two feet vs time to see the local maximums(positive peaks) and minimums(negative peaks) to find the stride events.


# Normalize x coordinates by linear interpolation alpha/(1-alpha) wrt x_0 x_n(n_th frame x diff) and draw the Y_diff vs X_diff with shifting the whole system with the first frame x diff and y diff
# Calculate winding factors and when each time the winding angle hit some time of 2pi then one stride is finished  should be instable for patients
# Determine the integral/area of the curve y_diff_norm vs frame for the aggregate info about the stride height and time for gait analysis
# Shoot a video of normal person walking from sideways and draw Y_diff vs x_diff and should find that the loops are more stable
# Compare pre-op and post-op
# For x_diff normalization, use xi to xj, denoting the xdiff at frame i and j where i is close to 0 j is close to n and take linear interpolation for all frames k
# Choose separation points for each stride and stack all strides hills together to see whether the stride pattern is consistent across strides
# For each stride, slice the curve vertically and calculate crossing points' variation, mean value and find a way to compute the difference areas between the strides curves
# Take the second derivatives of the patients curves and the normal person curve and stack them together. 
# For normal person, second derivatives should be more smooth and less peaky, but for patients, it should have a much higher amplitude and more peaky and inconsistent

