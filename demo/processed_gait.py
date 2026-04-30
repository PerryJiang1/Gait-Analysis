import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os


track_json_path = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Chiocchi_mirror\track.json"
out_dir = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Chiocchi_mirror\metrics"

# COCO indices
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


def load_track_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = []
    kpts = []
    scores = []

    for item in data:
        if item.get("person_idx") is None:
            continue
        if "keypoints" not in item:
            continue

        frames.append(item["frame"])
        kpts.append(np.array(item["keypoints"], dtype=np.float32))
        scores.append(np.array(item["scores"], dtype=np.float32))

    if len(kpts) == 0:
        raise ValueError("No valid tracked keypoints found in track.json")

    return np.array(frames), np.stack(kpts), np.stack(scores)


def euclidean_series(p1, p2):
    return np.linalg.norm(p1 - p2, axis=1)


def smooth_1d(y, window_length=31, polyorder=3):
    y = np.asarray(y, dtype=np.float32)
    n = len(y)

    # window_length must be odd and <= n
    if n < 5:
        return y.copy()

    wl = min(window_length, n if n % 2 == 1 else n - 1)
    if wl < 5:
        return y.copy()
    if wl % 2 == 0:
        wl -= 1

    polyorder = min(polyorder, wl - 1)
    return savgol_filter(y, window_length=wl, polyorder=polyorder)


def detrend_with_savgol(y, trend_window=201, trend_polyorder=3):
    """
    detrended = y - smooth_large_window(y)
    """
    trend = smooth_1d(y, window_length=trend_window, polyorder=trend_polyorder)
    detrended = y - trend
    return detrended, trend

def save_and_show(save_path):
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

def plot_knee_distance(frames, kpts):
    lknee = kpts[:, L_KNEE, :]
    rknee = kpts[:, R_KNEE, :]

    knee_x_dist = np.abs(lknee[:, 0] - rknee[:, 0])
    knee_euc_dist = euclidean_series(lknee, rknee)

    plt.figure(figsize=(14, 6))
    plt.plot(frames, knee_x_dist, label="abs(x_LKnee - x_RKnee)", linewidth=2)
    plt.plot(frames, knee_euc_dist, label="euclidean(LKnee, RKnee)", linewidth=2)
    plt.xlabel("Frame")
    plt.ylabel("Distance (pixels)")
    plt.title("Knee Distance vs Frame")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(out_dir, "01_knee_distance.png")
    save_and_show(save_path)


def plot_ankle_curves_for_swing(frames, kpts):
    lank_y = kpts[:, L_ANKLE, 1]
    rank_y = kpts[:, R_ANKLE, 1]

    lank_y_s = smooth_1d(lank_y, window_length=31, polyorder=3)
    rank_y_s = smooth_1d(rank_y, window_length=31, polyorder=3)

    lank_y_d, lank_trend = detrend_with_savgol(lank_y_s, trend_window=201, trend_polyorder=3)
    rank_y_d, rank_trend = detrend_with_savgol(rank_y_s, trend_window=201, trend_polyorder=3)

    # 1) raw ankle y
    plt.figure(figsize=(14, 6))
    plt.plot(frames, lank_y, label="Left ankle y (raw)", linewidth=1.5)
    plt.plot(frames, rank_y, label="Right ankle y (raw)", linewidth=1.5)
    plt.xlabel("Frame")
    plt.ylabel("Y coordinate")
    plt.title("Raw Ankle Y vs Frame")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_and_show(os.path.join(out_dir, "02_ankle_raw.png"))

    # 2) smoothed ankle y + trend
    plt.figure(figsize=(14, 6))
    plt.plot(frames, lank_y_s, label="Left ankle y (smoothed)", linewidth=2)
    plt.plot(frames, rank_y_s, label="Right ankle y (smoothed)", linewidth=2)
    plt.xlabel("Frame")
    plt.ylabel("Y coordinate")
    plt.title("Smoothed Ankle Y and Slow Trend")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_and_show(os.path.join(out_dir, "03_ankle_smoothed_trend.png"))

    # 3) detrended ankle y
    plt.figure(figsize=(14, 6))
    plt.plot(frames, lank_y_d, label="Left ankle y (detrended)", linewidth=2)
    plt.plot(frames, rank_y_d, label="Right ankle y (detrended)", linewidth=2)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Frame")
    plt.ylabel("Detrended Y")
    plt.title("Detrended Ankle Y vs Frame (for Swing/Symmetry Inspection)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_and_show(os.path.join(out_dir, "04_ankle_detrended.png"))

    return {
        "left_raw": lank_y,
        "right_raw": rank_y,
        "left_smooth": lank_y_s,
        "right_smooth": rank_y_s,
        "left_detrended": lank_y_d,
        "right_detrended": rank_y_d,
    }


def plot_hand_wrt_shoulder(frames, kpts):
    lshoulder_y = kpts[:, L_SHOULDER, 1]
    rshoulder_y = kpts[:, R_SHOULDER, 1]
    lwrist_y = kpts[:, L_WRIST, 1]
    rwrist_y = kpts[:, R_WRIST, 1]

    left_hand_shoulder_ydist = lwrist_y - lshoulder_y
    right_hand_shoulder_ydist = rwrist_y - rshoulder_y

    plt.figure(figsize=(14, 6))
    plt.plot(frames, left_hand_shoulder_ydist, label="left_wrist_y - left_shoulder_y", linewidth=2)
    plt.plot(frames, right_hand_shoulder_ydist, label="right_wrist_y - right_shoulder_y", linewidth=2)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Frame")
    plt.ylabel("Vertical distance in y (pixels)")
    plt.title("Wrist Vertical Position w.r.t. Shoulder vs Frame")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(out_dir, "05_wrist_shoulder_ydist.png")
    save_and_show(save_path)


if __name__ == "__main__":
    os.makedirs(out_dir, exist_ok=True)
    frames, kpts, scores = load_track_json(track_json_path)

    # 1. Knee distance vs frame
    plot_knee_distance(frames, kpts)

    # 2. Ankle curves for swing phase symmetry
    ankle_dict = plot_ankle_curves_for_swing(frames, kpts)

    # 3. Hand position wrt shoulder
    plot_hand_wrt_shoulder(frames, kpts)