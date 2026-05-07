import os
import json
import glob
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from classes_and_palettes import (
    COCO_KPTS_COLORS,
    COCO_WHOLEBODY_KPTS_COLORS,
    GOLIATH_KPTS_COLORS,
    GOLIATH_SKELETON_INFO,
    COCO_SKELETON_INFO,
    COCO_WHOLEBODY_SKELETON_INFO
)
from tqdm import tqdm


KPT_THR = 0.3

def draw_one_skeleton(img, kpts, scores, kpt_thr, radius, thickness, kpt_colors, skeleton_info):
    kpts = np.asarray(kpts)
    scores = np.asarray(scores)

    # draw keypoints
    for kid, (x, y) in enumerate(kpts):
        if scores[kid] < kpt_thr:
            continue
        color = kpt_colors[kid]
        if not isinstance(color, str):
            color = tuple(int(c) for c in color[::-1])
        cv2.circle(img, (int(x), int(y)), int(radius), color, -1)

    for _, link_info in skeleton_info.items():
        a, b = link_info["link"]
        if scores[a] < kpt_thr or scores[b] < kpt_thr:
            continue
        color = link_info["color"][::-1]
        pt1 = (int(kpts[a][0]), int(kpts[a][1]))
        pt2 = (int(kpts[b][0]), int(kpts[b][1]))
        cv2.line(img, pt1, pt2, color, thickness=thickness)

    return img

def load_frame_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    persons = []
    for inst in data["instance_info"]:
        kpts = np.array(inst["keypoints"], dtype=np.float32)
        conf_scores  = np.array(inst["keypoint_scores"], dtype=np.float32)
        persons.append((kpts, conf_scores))
    return persons

def person_features(kpts, conf_scores, kpt_thr=KPT_THR):
    vis = conf_scores >= kpt_thr
    if vis.sum() < 4:
        vis = conf_scores >= (kpt_thr * 0.5)

    pts = kpts[vis]
    if len(pts) == 0:
        pts = kpts

    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    w, h = max(1.0, x2-x1), max(1.0, y2-y1)
    cx, cy = x1 + w/2, y1 + h/2
    scale = np.sqrt(w*w + h*h)

    norm = (kpts - np.array([cx, cy], dtype=np.float32)) / max(scale, 1.0)
    emb = norm.reshape(-1)

    quality = float(conf_scores.mean())
    return {
        "bbox": np.array([x1,y1,x2,y2], dtype=np.float32),
        "center": np.array([cx,cy], dtype=np.float32),
        "scale": float(scale),
        "emb": emb,
        "scores": conf_scores,
        "kpts": kpts,
        "quality": quality
    }

def cost_between(a, b, img_diag, w_pose=0.6, w_center=0.2, w_quality=0.1, w_scale=0.1,):
    dc = np.linalg.norm(a["center"] - b["center"]) / img_diag
    dp = np.linalg.norm(a["emb"] - b["emb"]) / np.sqrt(len(a["emb"]))

    sa = max(float(a["scale"]), 1e-6)
    sb = max(float(b["scale"]), 1e-6)
    ds = abs(np.log(sb / sa))

    # if dc > 0.25:
    #     return 1e9
    # if dp > 0.5:
    #     return 1e9
    # if ds > 1.0:
    #     return 1e9

    dq = -(b["quality"] - a["quality"])
    return w_pose*dp + w_center*dc + w_quality*dq + w_scale * ds

def draw_skeleton_simple(img, kpts, conf_scores, color=(0,255,0), kpt_thr=KPT_THR):
    for (x,y), s in zip(kpts, conf_scores):
        if s >= kpt_thr:
            cv2.circle(img, (int(x),int(y)), 3, color, -1)
    return img

def choose_target_ui(img, feats_list, max_w=1400, max_h=900):
    vis = img.copy()

    # 1) draw bbox + id + skeleton
    for i, f in enumerate(feats_list):
        x1, y1, x2, y2 = f["bbox"].astype(int)
        cx, cy = f["center"]

        # bbox rectangle (black)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0,0,0), 2)

        # id text near bbox top-left (black)
        cv2.putText(vis, f"ID {i}", (x1, max(0, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

        # optional: draw center point
        cv2.circle(vis, (int(cx), int(cy)), 3, (0,0,0), -1)

        # draw keypoints (green) for reference
        draw_skeleton_simple(vis, f["kpts"], f["scores"], color=(0,255,0))

    # instruction text (black)
    cv2.putText(vis, "Click bbox to select | Press 's' to SKIP | Press 'q' to QUIT",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    
    H, W = vis.shape[:2]
    scale = min(max_w / W, max_h / H, 1.0)
    if scale < 1.0:
        vis_show = cv2.resize(vis, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
    else:
        vis_show = vis

    chosen = {"idx": None, "skip": False, "quit": False}


    def pick_by_bbox(x, y):
        """Return best idx whose bbox contains (x,y); if multiple, choose smallest area."""
        hits = []
        for i, f in enumerate(feats_list):
            x1, y1, x2, y2 = f["bbox"]
            if x >= x1 and x <= x2 and y >= y1 and y <= y2:
                area = float((x2 - x1) * (y2 - y1))
                hits.append((area, i))
        if not hits:
            return None
        hits.sort()  # smallest area first
        return hits[0][1]

    def pick_by_center(x, y):
        p = np.array([x, y], dtype=np.float32)
        d = [np.linalg.norm(f["center"] - p) for f in feats_list]
        return int(np.argmin(d)) if len(d) else None

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x0 = float(x) / scale
            y0 = float(y) / scale
            idx = pick_by_bbox(x0, y0)
            if idx is None:
                idx = pick_by_center(x, y)
            chosen["idx"] = idx

    cv2.namedWindow("select", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("select", on_mouse)

    while True:
        cv2.imshow("select", vis_show)
        key = cv2.waitKey(10) & 0xFF
        if chosen["idx"] is not None:
            break
        if key == ord('s'):
            chosen["skip"] = True
            break
        if key == ord('q'):
            chosen["quit"] = True
            break

    cv2.destroyWindow("select")

    if chosen.get("quit", False):
        return -2   # quit
    if chosen["skip"]:
        return -1   # skip
    return chosen["idx"]

def draw_skeleton_color(img, kpts, scores, skeleton_info, kpt_thr=0.10, radius=6, thickness=3, bgr=(0,0,255)):
    """Draw keypoints+links in a single color (BGR)."""
    kpts = np.asarray(kpts)
    scores = np.asarray(scores)

    # draw keypoints (single color)
    for kid, (x, y) in enumerate(kpts):
        if scores[kid] < kpt_thr:
            continue
        cv2.circle(img, (int(x), int(y)), int(radius), bgr, -1)

    # draw links (single color)
    for _, link_info in skeleton_info.items():
        a, b = link_info["link"]
        if scores[a] < kpt_thr or scores[b] < kpt_thr:
            continue
        pt1 = (int(kpts[a][0]), int(kpts[a][1]))
        pt2 = (int(kpts[b][0]), int(kpts[b][1]))
        cv2.line(img, pt1, pt2, bgr, thickness=thickness)

    return img


def confirm_candidate_ui(img, feats_list, cand_idx, max_w=1400, max_h=900, kpt_thr=0.10):
    """
    Show current frame with all bbox+ID in black, and candidate skeleton highlighted in RED.
    Keys:
      y: accept candidate
      n: manual reselect (mouse)
      s: skip this frame
      q: quit pipeline
    Returns: one of {"y","n","s","q"}
    """
    vis = img.copy()

    # draw all bbox + id in black
    for i, f in enumerate(feats_list):
        x1, y1, x2, y2 = f["bbox"].astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0,0,0), 2)
        cv2.putText(vis, f"ID {i}", (x1, max(0, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

    # highlight candidate in red
    if cand_idx is not None and 0 <= cand_idx < len(feats_list):
        f = feats_list[cand_idx]
        x1, y1, x2, y2 = f["bbox"].astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0,0,255), 3)
        draw_skeleton_color(vis, f["kpts"], f["scores"], skeleton_info=skeleton_info, kpt_thr=kpt_thr, radius=6, thickness=3, bgr=(0,0,255))

    cv2.putText(vis, "Candidate (RED). Press y=accept | n=manual | s=skip | q=quit",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

    H, W = vis.shape[:2]
    scale = min(max_w / W, max_h / H, 1.0)
    if scale < 1.0:
        vis_show = cv2.resize(vis, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
    else:
        vis_show = vis

    cv2.namedWindow("confirm", cv2.WINDOW_NORMAL)

    while True:
        cv2.imshow("confirm", vis_show)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('y'):
            cv2.destroyWindow("confirm")
            return "y"
        if key == ord('n'):
            cv2.destroyWindow("confirm")
            return "n"
        if key == ord('s'):
            cv2.destroyWindow("confirm")
            return "s"
        if key == ord('q'):
            cv2.destroyWindow("confirm")
            return "q"

# def track_sequence_to_jpg(
#     json_dir,
#     img_dir,
#     out_track_path,
#     out_dir,
#     img_ext="jpg",
#     kpt_thr=KPT_THR,
#     radius=6,
#     thickness=3,
#     kpt_colors=None,
#     skeleton_info=None,
# ):
#     os.makedirs(out_dir, exist_ok=True)

#     json_paths = sorted(glob.glob(f"{json_dir}/*.json"))
#     assert len(json_paths) > 0

#     # first frame image to get diag
#     first_img_path = os.path.join(img_dir, os.path.splitext(os.path.basename(json_paths[0]))[0] + f".{img_ext}")
#     img0 = cv2.imread(first_img_path)
#     if img0 is None:
#         raise FileNotFoundError(first_img_path)

#     H, W = img0.shape[:2]
#     img_diag = float(np.sqrt(H*H + W*W))

#     # first frame candidates
#     persons0 = load_frame_json(json_paths[0])
#     feats0 = [person_features(k,s, kpt_thr=kpt_thr) for k,s in persons0]

#     target_idx = choose_target_ui(img0, feats0)
#     if target_idx is None:
#         print("No target selected.")
#         return

#     prev_feat = feats0[target_idx]

#     track = []
#     track.append({
#         "frame": 0,
#         "person_idx": int(target_idx),
#         "keypoints": prev_feat["kpts"].tolist(),
#         "scores": prev_feat["scores"].tolist()
#     })

#     out0 = img0.copy()
#     out0 = draw_one_skeleton(out0, prev_feat["kpts"], prev_feat["scores"],
#                              kpt_thr=kpt_thr, radius=radius, thickness=thickness,
#                              kpt_colors=kpt_colors, skeleton_info=skeleton_info)
#     stem0 = os.path.splitext(os.path.basename(json_paths[0]))[0]
#     cv2.imwrite(os.path.join(out_dir, stem0 + f".{img_ext}"), out0)

#     # iterate next frames
#     for t in range(1, len(json_paths)):
#         stem = os.path.splitext(os.path.basename(json_paths[t]))[0]
#         img_path = os.path.join(img_dir, stem + f".{img_ext}")
#         img = cv2.imread(img_path)

#         if img is None:
#             print("skip (no image):", img_path)
#             track.append({"frame": t, "person_idx": None})
#             continue

#         persons = load_frame_json(json_paths[t])
#         feats = [person_features(k,s, kpt_thr=kpt_thr) for k,s in persons]

#         out_img = img.copy()

#         if len(feats) == 0:
#             cv2.imwrite(os.path.join(out_dir, stem + f".{img_ext}"), out_img)
#             track.append({"frame": t, "person_idx": None})
#             continue

#         costs = [cost_between(prev_feat, f, img_diag) for f in feats]
#         j = int(np.argmin(costs))

#         if costs[j] >= 1e8:
#             cv2.imwrite(os.path.join(out_dir, stem + f".{img_ext}"), out_img)
#             track.append({"frame": t, "person_idx": None})
#             continue

#         prev_feat = feats[j]
#         out_img = draw_one_skeleton(out_img, prev_feat["kpts"], prev_feat["scores"],
#                                     kpt_thr=kpt_thr, radius=radius, thickness=thickness,
#                                     kpt_colors=kpt_colors, skeleton_info=skeleton_info)

#         cv2.imwrite(os.path.join(out_dir, stem + f".{img_ext}"), out_img)

#         track.append({
#             "frame": t,
#             "person_idx": int(j),
#             "keypoints": prev_feat["kpts"].tolist(),
#             "scores": prev_feat["scores"].tolist()
#         })

#     with open(out_track_path, "w", encoding="utf-8") as f:
#         json.dump(track, f, ensure_ascii=False, indent=2)

#     print("saved tracked jpgs to:", out_dir)
#     print("saved track json to:", out_track_path)

def bipartite_assign(feats_prev, feats_cur, img_diag, unmatch_cost=1e6):
    """
    Hungarian assignment between all persons in prev frame and current frame.
    Returns:
      match_prev_to_cur: list length len(feats_prev), value is matched j or None
      match_costs: list length len(feats_prev), value is cost or None
    """
    n_prev = len(feats_prev)
    n_cur  = len(feats_cur)
    if n_prev == 0:
        return [], []

    n = max(n_prev, n_cur)
    C = np.full((n, n), unmatch_cost, dtype=np.float64)

    # fill real costs
    for i in range(n_prev):
        for j in range(n_cur):
            c = cost_between(feats_prev[i], feats_cur[j], img_diag)
            C[i, j] = c

    # Hungarian on square matrix
    row_ind, col_ind = linear_sum_assignment(C)

    match_prev_to_cur = [None] * n_prev
    match_costs = [None] * n_prev
    for r, c in zip(row_ind, col_ind):
        if r < n_prev:
            if c < n_cur and C[r, c] < unmatch_cost:
                # matched to a real detection
                match_prev_to_cur[r] = int(c)
                match_costs[r] = float(C[r, c])
            else:
                # matched to dummy -> unmatched
                match_prev_to_cur[r] = None
                match_costs[r] = None

    return match_prev_to_cur, match_costs

def build_cost_matrix(feats_prev, feats_cur, img_diag, unmatch_cost=1e6):
    """
    Build a padded square cost matrix for Hungarian.
    Returns:
      C_pad: (n,n) padded cost matrix
      C_real: (n_prev, n_cur) real cost matrix
    """
    n_prev = len(feats_prev)
    n_cur = len(feats_cur)
    n = max(n_prev, n_cur)
    C_pad = np.full((n, n), unmatch_cost, dtype=np.float64)
    C_real = np.full((n_prev, n_cur), unmatch_cost, dtype=np.float64)

    for i in range(n_prev):
        for j in range(n_cur):
            c = cost_between(feats_prev[i], feats_cur[j], img_diag)
            C_real[i, j] = c
            C_pad[i, j] = c

    return C_pad, C_real


def track_sequence_to_jpg_hungarian_reselect(
    json_dir,
    img_dir,
    out_track_path,
    out_edges_path,
    out_dir,
    img_ext="jpg",
    kpt_thr=KPT_THR,
    viz_kpt_thr=0.10,
    radius=6,
    thickness=3,
    kpt_colors=None,
    skeleton_info=None,
    unmatch_cost=1e6,
    match_cost_thr=1e5
):
    os.makedirs(out_dir, exist_ok=True)

    json_paths = sorted(glob.glob(os.path.join(json_dir, "frame_*.json")))
    assert len(json_paths) > 0

    track = []
    edges = []

    def dump_outputs():
        os.makedirs(os.path.dirname(out_track_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(out_edges_path) or ".", exist_ok=True)
        with open(out_track_path, "w", encoding="utf-8") as f:
            json.dump(track, f, ensure_ascii=False, indent=2)
        with open(out_edges_path, "w", encoding="utf-8") as f:
            json.dump(edges, f, ensure_ascii=False, indent=2)
        print("saved track json to:", out_track_path)
        print("saved edges json to:", out_edges_path)

    # -------- frame 0 --------
    stem0 = os.path.splitext(os.path.basename(json_paths[0]))[0]
    img0_path = os.path.join(img_dir, stem0 + f".{img_ext}")
    img0 = cv2.imread(img0_path)
    if img0 is None:
        raise FileNotFoundError(img0_path)

    H, W = img0.shape[:2]
    img_diag = float(np.sqrt(H*H + W*W))

    persons0 = load_frame_json(json_paths[0])
    feats0 = [person_features(k, s, kpt_thr=kpt_thr) for k, s in persons0]
    confirm_next_match = False

    if len(feats0) == 0:
        # no people in frame0, save empty and anchor stays None
        cv2.imwrite(os.path.join(out_dir, stem0 + f".{img_ext}"), img0)
        track.append({"frame": 0, "person_idx": None})
        feats_anchor = None
        anchor_target_idx = None
        anchor_frame = None
    else:
        idx0 = choose_target_ui(img0, feats0)
        if idx0 == -2:
            print("Quit by user (frame 0).")
            dump_outputs()
            return
        if idx0 == -1:
            # user skipped frame0 (allowed)
            cv2.imwrite(os.path.join(out_dir, stem0 + f".{img_ext}"), img0)
            track.append({"frame": 0, "person_idx": None})
            feats_anchor = None
            anchor_target_idx = None
            anchor_frame = None
            confirm_next_match = True
        else:
            idx0 = int(idx0)
            feat0 = feats0[idx0]
            out0 = draw_one_skeleton(
                img0.copy(), feat0["kpts"], feat0["scores"],
                kpt_thr=viz_kpt_thr, radius=radius, thickness=thickness,
                kpt_colors=kpt_colors, skeleton_info=skeleton_info
            )
            cv2.imwrite(os.path.join(out_dir, stem0 + f".{img_ext}"), out0)
            track.append({
                "frame": 0, "person_idx": idx0,
                "keypoints": feat0["kpts"].tolist(),
                "scores": feat0["scores"].tolist()
            })
            # set anchor
            feats_anchor = feats0
            anchor_target_idx = idx0
            anchor_frame = 0

    # -------- iterate frames 1.. --------
    for t in tqdm(range(1, len(json_paths)), total=len(json_paths)-1, desc="Tracking frames"):
        stem = os.path.splitext(os.path.basename(json_paths[t]))[0]
        img_path = os.path.join(img_dir, stem + f".{img_ext}")
        img = cv2.imread(img_path)

        if img is None:
            print("skip (no image):", img_path)
            track.append({"frame": t, "person_idx": None})
            continue

        persons = load_frame_json(json_paths[t])
        feats_cur = [person_features(k, s, kpt_thr=kpt_thr) for k, s in persons]

        # store cost matrix edges vs anchor if anchor exists (matches你的“i-1 vs i+1 / i+2 ...”)
        if feats_anchor is not None and anchor_target_idx is not None:
            _, C_real = build_cost_matrix(feats_anchor, feats_cur, img_diag, unmatch_cost=unmatch_cost)
            edges.append({
                "anchor_frame": int(anchor_frame),
                "frame_cur": int(t),
                "stem_anchor": os.path.splitext(os.path.basename(json_paths[anchor_frame]))[0],
                "stem_cur": stem,
                "cost_matrix": C_real.tolist()
            })

        out_img = img.copy()

        # no detections -> cannot match; skip frame, anchor unchanged
        if len(feats_cur) == 0:
            cv2.imwrite(os.path.join(out_dir, stem + f".{img_ext}"), out_img)
            track.append({"frame": t, "person_idx": None})
            continue

        # If no anchor yet, must ask user (or allow skip/quit)
        if feats_anchor is None or anchor_target_idx is None:
            idx = choose_target_ui(img, feats_cur)
            if idx == -2:
                print(f"Quit by user (frame {t}).")
                dump_outputs()
                return
            if idx == -1:
                cv2.imwrite(os.path.join(out_dir, stem + f".{img_ext}"), out_img)
                track.append({"frame": t, "person_idx": None})
                confirm_next_match = True
                continue
            idx = int(idx)
            feat = feats_cur[idx]
            out_img = draw_one_skeleton(
                out_img, feat["kpts"], feat["scores"],
                kpt_thr=viz_kpt_thr, radius=radius, thickness=thickness,
                kpt_colors=kpt_colors, skeleton_info=skeleton_info
            )
            cv2.imwrite(os.path.join(out_dir, stem + f".{img_ext}"), out_img)
            track.append({
                "frame": t, "person_idx": idx,
                "keypoints": feat["kpts"].tolist(),
                "scores": feat["scores"].tolist()
            })
            feats_anchor = feats_cur
            anchor_target_idx = idx
            anchor_frame = t
            continue

        # -------- anchor-based Hungarian proposal --------
        match_prev_to_cur, match_costs = bipartite_assign(
            feats_anchor, feats_cur, img_diag, unmatch_cost=unmatch_cost
        )

        cand_idx = None
        cand_cost = None
        if anchor_target_idx < len(match_prev_to_cur):
            cand_idx = match_prev_to_cur[anchor_target_idx]
            cand_cost = match_costs[anchor_target_idx]

        # if unmatched or too large cost -> still propose (red) if we have a candidate; else go manual path
        if cand_idx is None or cand_cost is None or cand_cost >= match_cost_thr:
            # No good candidate from Hungarian, ask user to choose (or skip/quit)
            idx = choose_target_ui(img, feats_cur)
            if idx == -2:
                print(f"Quit by user (frame {t}).")
                dump_outputs()
                return
            if idx == -1:
                cv2.imwrite(os.path.join(out_dir, stem + f".{img_ext}"), out_img)
                track.append({"frame": t, "person_idx": None})
                confirm_next_match = True
                # anchor unchanged
                continue
            idx = int(idx)
            feat = feats_cur[idx]
            out_img = draw_one_skeleton(
                out_img, feat["kpts"], feat["scores"],
                kpt_thr=viz_kpt_thr, radius=radius, thickness=thickness,
                kpt_colors=kpt_colors, skeleton_info=skeleton_info
            )
            cv2.imwrite(os.path.join(out_dir, stem + f".{img_ext}"), out_img)
            track.append({
                "frame": t, "person_idx": idx,
                "keypoints": feat["kpts"].tolist(),
                "scores": feat["scores"].tolist()
            })
            feats_anchor = feats_cur
            anchor_target_idx = idx
            anchor_frame = t
            continue

        # -------- show candidate in red and ask y/n/s/q --------
        if confirm_next_match:
            action = confirm_candidate_ui(img, feats_cur, cand_idx, kpt_thr=viz_kpt_thr)

            if action == "q":
                print(f"Quit by user (frame {t}).")
                dump_outputs()
                return

            if action == "s":
                cv2.imwrite(os.path.join(out_dir, stem + f".{img_ext}"), out_img)
                track.append({"frame": t, "person_idx": None})
                # anchor unchanged, next frame will match anchor -> (t+1)
                confirm_next_match = True
                continue

            if action == "n":
                idx = choose_target_ui(img, feats_cur)
                if idx == -2:
                    print(f"Quit by user (frame {t}).")
                    dump_outputs()
                    return
                if idx == -1:
                    cv2.imwrite(os.path.join(out_dir, stem + f".{img_ext}"), out_img)
                    track.append({"frame": t, "person_idx": None})
                    confirm_next_match = True
                    continue
                idx = int(idx)
                feat = feats_cur[idx]
                out_img = draw_one_skeleton(
                    out_img, feat["kpts"], feat["scores"],
                    kpt_thr=viz_kpt_thr, radius=radius, thickness=thickness,
                    kpt_colors=kpt_colors, skeleton_info=skeleton_info
                )
                cv2.imwrite(os.path.join(out_dir, stem + f".{img_ext}"), out_img)
                track.append({
                    "frame": t, "person_idx": idx,
                    "keypoints": feat["kpts"].tolist(),
                    "scores": feat["scores"].tolist()
                })
                feats_anchor = feats_cur
                anchor_target_idx = idx
                anchor_frame = t
                continue
            confirm_next_match = False

        # action == "y": accept candidate
        feat = feats_cur[cand_idx]
        out_img = img.copy()
        out_img = draw_one_skeleton(
            out_img, feat["kpts"], feat["scores"],
            kpt_thr=viz_kpt_thr, radius=radius, thickness=thickness,
            kpt_colors=kpt_colors, skeleton_info=skeleton_info
        )

        cv2.imwrite(os.path.join(out_dir, stem + f".{img_ext}"), out_img)
        track.append({
            "frame": t, "person_idx": int(cand_idx),
            "keypoints": feat["kpts"].tolist(),
            "scores": feat["scores"].tolist()
        })
        feats_anchor = feats_cur
        anchor_target_idx = int(cand_idx)
        anchor_frame = t

    dump_outputs()



if __name__ == "__main__":
    # json_dir = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Sronce_walker"
    # img_dir  = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\data\Sronce_walker"
    # out_dir  = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Sronce_walker\tracked"

    # json_dir = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Ovcharenko_preop"
    # img_dir  = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\data\Ovcharenko_preop"
    # out_dir  = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Ovcharenko_preop\tracked"

    # json_dir = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\downtown_cafe\17"
    # img_dir  = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\data\downtown_cafe_00"
    # out_dir  = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\downtown_cafe\tracked"

    # json_dir = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Chiocchi_mirror"
    # img_dir  = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\data\Chiocchi_mirror"
    # out_dir  = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Chiocchi_mirror"

    # json_dir = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Baseline"
    # img_dir  = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\data\Baseline"
    # out_dir  = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Baseline\tracked"

    json_dir = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Sronce_preop"
    img_dir  = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\data\Sronce_preop"
    out_dir  = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Sronce_preop\tracked"

    # json_dir = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Baseline_side"
    # img_dir  = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\data\Baseline_side"
    # out_dir  = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\output\pose\Baseline_side\tracked"
    out_track_path = os.path.join(out_dir, "track.json")
    out_edges_path = os.path.join(out_dir, "edges_costs.json")

    kpt_colors = COCO_KPTS_COLORS
    skeleton_info = COCO_SKELETON_INFO

    track_sequence_to_jpg_hungarian_reselect(
        json_dir=json_dir,
        img_dir=img_dir,
        out_dir=out_dir,
        out_track_path=out_track_path,
        out_edges_path=out_edges_path,
        img_ext="jpg",
        kpt_thr=0.3,
        viz_kpt_thr=0.10,
        radius=6,
        thickness=3,
        kpt_colors=kpt_colors,
        skeleton_info=skeleton_info,
        unmatch_cost=1e6,
        match_cost_thr=1e5
    )

# left/right swing phase symmetry/ 
# knees distance / 
# knee to ankle distance wrt height/ 
# swing phase time/ 
# hand position with shoulder/ 
# heel distance to ground