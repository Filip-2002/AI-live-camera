import argparse
import time
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

import os

def color_for(text):
    import hashlib, random
    h = int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16) & 0xFFFFFFFF
    rng = random.Random(h)
    return (rng.randint(30, 255), rng.randint(30, 255), rng.randint(30, 255))

def collect_detections(res):
    """
    From a single Ultralytics Results object, return:
      boxes_n: [[x1,y1,x2,y2] normalized 0-1]
      scores:  [conf...]
      names:   [class_name...] (lowercased for unification)
    """
    if res is None or res.boxes is None or len(res.boxes) == 0:
        return [], [], []

    boxes_n = res.boxes.xyxyn.cpu().numpy().tolist()
    scores = res.boxes.conf.cpu().numpy().tolist()
    cls_idx = res.boxes.cls.cpu().numpy().astype(int).tolist()

    id2name = res.names if hasattr(res, "names") else {}
    names = [str(id2name.get(int(c), f"class_{int(c)}")).lower() for c in cls_idx]
    return boxes_n, scores, names

def draw_fused(frame, fused_boxes_n, fused_scores, fused_label_ids, id2name):
    H, W = frame.shape[:2]
    for box, score, lid in zip(fused_boxes_n, fused_scores, fused_label_ids):
        x1, y1, x2, y2 = box
        p1 = (int(x1 * W), int(y1 * H))
        p2 = (int(x2 * W), int(y2 * H))
        name = id2name.get(int(lid), str(lid))
        color = color_for(name)

        cv2.rectangle(frame, p1, p2, color, 2)
        label = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), color, -1)
        cv2.putText(frame, label, (p1[0] + 2, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    return frame 

def main():
    ap = argparse.ArgumentParser(description="YOLOv8 dual-model webcam with Weighted Box Fusion")
    ap.add_argument("--coco", default="yolov8l.pt", help="Path to COCO model weights")
    ap.add_argument("--oiv7", default="yolov8l-oiv7.pt", help="Path to OIV7 model weights")
    ap.add_argument("--source", default="0", help="Webcam index or video filename (looked up in 'videos/' folder)")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    ap.add_argument("--conf", type=float, default=0.25, help="Per-model confidence threshold")
    ap.add_argument("--iou", type=float, default=0.7, help="Per-model NMS IoU")
    ap.add_argument("--wbf_iou", type=float, default=0.55, help="WBF IoU threshold")
    ap.add_argument("--skip_box_thr", type=float, default=0.001, help="WBF skip threshold")
    ap.add_argument("--device", default=None, help="torch device, e.g. 'cpu' or 'cuda:0'")
    ap.add_argument("--save", action="store_true", help="Save output video to videos/outputs/ folder")
    args = ap.parse_args()

    print("Loading models...")
    model_coco = YOLO(args.coco)
    model_oiv7 = YOLO(args.oiv7)

    if args.source.isdigit():
        src = int(args.source)
    else:
        src = os.path.join("videos", args.source)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {src}")

    out_writer = None
    if args.save:
        os.makedirs("videos/outputs", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = os.path.join("videos/outputs", "output.mp4")
        out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"Saving output video to {out_path}")

    name2id = {}
    id2name = {}

    def get_unified_id(name_lower):
        if name_lower not in name2id:
            new_id = len(name2id)
            name2id[name_lower] = new_id
            id2name[new_id] = name_lower
        return name2id[name_lower]

    fps_avg = None
    t_prev = time.time()

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res1 = model_coco.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]
        res2 = model_oiv7.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]

        b1, s1, n1 = collect_detections(res1)
        b2, s2, n2 = collect_detections(res2)

        l1 = [get_unified_id(nm) for nm in n1]
        l2 = [get_unified_id(nm) for nm in n2]

        boxes_list = [b1, b2]
        scores_list = [s1, s2]
        labels_list = [l1, l2]

        if (not b1) and (not b2):
            fused_boxes, fused_scores, fused_labels = [], [], []
        else:
            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                iou_thr=args.wbf_iou,
                skip_box_thr=args.skip_box_thr,
                conf_type="avg"
            )

        out = frame.copy()
        out = draw_fused(out, fused_boxes, fused_scores, fused_labels, id2name)

        now = time.time()
        dt = now - t_prev
        t_prev = now
        fps = 1.0 / dt if dt > 0 else 0.0
        fps_avg = fps if fps_avg is None else (0.9 * fps_avg + 0.1 * fps)
        cv2.putText(out, f"FPS: {fps_avg:.1f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (25, 255, 25), 2, cv2.LINE_AA)

        cv2.imshow("YOLOv8 COCO + OIV7 (WBF fused)", out)

        if out_writer:
            out_writer.write(out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
