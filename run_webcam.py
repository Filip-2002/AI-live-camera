import argparse
import yaml
import cv2
import torch
import numpy as np

from detector import YoloDetector
from tracker import IOUTracker
from zones import ZoneManager
from alerting import Alerter

COLORS = {}

def color_for(id):
    import random
    if id not in COLORS:
        COLORS[id] = (int(64 + random.random() * 191),
                      int(64 + random.random() * 191),
                      int(64 + random.random() * 191))
    return COLORS[id]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--model', default=None, help='YOLO model path')
    ap.add_argument('--source', default=0, help='webcam index or video path')
    ap.add_argument('--show', action='store_true')
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    if args.model:
        cfg['model_path'] = args.model

    det = YoloDetector(cfg)

    trk = IOUTracker(max_age=cfg['tracker']['max_age'],
                     thr=cfg['tracker']['iou_match_threshold'])
    zones = ZoneManager(cfg.get('zones', []))
    alerter = Alerter(cfg)

    cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else args.source)
    assert cap.isOpened(), f"Cannot open source {args.source}"

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        dets, infer_dt = det.infer(frame)

        tracker_inputs = dets
        tracks = trk.update(tracker_inputs)

        events = []
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.box)
            label = det.label(t.cls)
            zones_hit = zones.check(t.box, frame.shape)
            if zones_hit and t.cls in alerter.targets:
                events.append({"track": t.id, "class": t.cls, "label": label, "zones": zones_hit})

            cv2.rectangle(frame, (x1, y1), (x2, y2), color_for(t.id), 2)
            cv2.putText(frame, f"{label}#{t.id}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_for(t.id), 2)

        zones.draw(frame)
        alerter.maybe_alert(events)

        if args.show:
            cv2.putText(frame, f"FPS: {1.0 / max(1e-3, infer_dt):.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('AI Camera', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()



