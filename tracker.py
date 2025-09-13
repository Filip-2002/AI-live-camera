def iou(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2]-a[0]) * (a[3]-a[1])
    areaB = (b[2]-b[0]) * (b[3]-b[1])
    union = areaA + areaB - inter + 1e-6
    return inter / union

class Track:
    def __init__(self, tid, box, cls):
        self.id = tid
        self.box = box
        self.cls = cls
        self.age = 0
        self.hits = 1

class IOUTracker:
    def __init__(self, max_age=15, thr=0.3):
        self.max_age = max_age
        self.thr = thr
        self.next_id = 1
        self.tracks = []

    def update(self, detections):
        assigned = set()
        for t in self.tracks:
            best = -1; best_iou = 0
            for i, d in enumerate(detections):
                if i in assigned:
                    continue
                iouv = iou(t.box, d[:4])
                if iouv > best_iou:
                    best_iou = iouv; best = i
            if best_iou >= self.thr and best >= 0:
                t.box = detections[best][:4]
                t.cls = int(detections[best][5])
                t.age = 0
                t.hits += 1
                assigned.add(best)
            else:
                t.age += 1
        for i, d in enumerate(detections):
            if i not in assigned:
                self.tracks.append(Track(self.next_id, d[:4], int(d[5])))
                self.next_id += 1
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return self.tracks
