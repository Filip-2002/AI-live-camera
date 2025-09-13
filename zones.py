import numpy as np
import cv2

class ZoneManager:
    def __init__(self, zones):
        self.zones = zones or []

    @staticmethod
    def _point_in_poly(x, y, poly):
        inside = False
        n = len(poly)
        px, py = x, y
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1) % n]
            if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-9) + x1):
                inside = not inside
        return inside

    def check(self, box, frame_shape):
        h, w = frame_shape[:2]
        cx = (box[0] + box[2]) / 2 / w
        cy = (box[1] + box[3]) / 2 / h
        hits = []
        for z in self.zones:
            if self._point_in_poly(cx, cy, z['polygon']):
                hits.append(z['name'])
        return hits

    def draw(self, frame):
        h, w = frame.shape[:2]
        for z in self.zones:
            pts = np.array([[int(x*w), int(y*h)] for x, y in z['polygon']], dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 0), thickness=2)
