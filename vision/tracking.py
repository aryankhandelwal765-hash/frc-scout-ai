"""
Tracking: assign a persistent ID to each detection across frames.

Greedy nearest-centroid matching with a GRACE PERIOD -- a track survives a few
missed frames before being deleted. That matters here because:
  - robots cross paths and briefly occlude each other, and
  - a ball is constantly occluded while a robot is carrying it.

The tracker also reports DELETIONS (a track that went missing too long), which
is the signal the scoring engine uses: "a ball vanished -- where was it last?"
"""


class Track:
    __slots__ = ("id", "centroid", "bbox", "missed", "age", "visible", "history")

    def __init__(self, tid, centroid, bbox):
        self.id = tid
        self.centroid = centroid       # last known (cx, cy)
        self.bbox = bbox               # last known (x, y, w, h)
        self.missed = 0                # consecutive frames unmatched
        self.age = 0                   # total frames alive
        self.visible = True            # matched to a detection THIS frame?
        self.history = [centroid]      # centroid trail


class CentroidTracker:
    def __init__(self, max_distance=80.0, grace_frames=8):
        self.max_distance = max_distance
        self.grace_frames = grace_frames
        self.next_id = 0
        self.tracks = {}               # id -> Track

    def update(self, detections):
        """Match `detections` (list of {'centroid','bbox',...}) to existing tracks.

        Returns (live_tracks, deleted_tracks):
          live_tracks   -- dict id -> Track for every track still alive (incl.
                           ones coasting through the grace period; check
                           .visible to know if it was actually seen this frame).
          deleted_tracks -- list of Track objects removed this frame (missed too
                           long). Their .centroid/.bbox hold the last known spot.
        """
        # Reset per-frame visibility.
        for t in self.tracks.values():
            t.visible = False

        unmatched_track_ids = set(self.tracks.keys())

        # Greedy: build all (distance, det_idx, track_id) candidates under the
        # gate, sort, and claim closest-first so each is used at most once.
        candidates = []
        for di, det in enumerate(detections):
            cx, cy = det["centroid"]
            for tid, t in self.tracks.items():
                px, py = t.centroid
                dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if dist <= self.max_distance:
                    candidates.append((dist, di, tid))
        candidates.sort(key=lambda c: c[0])

        matched_dets = set()
        for dist, di, tid in candidates:
            if di in matched_dets or tid not in unmatched_track_ids:
                continue
            det = detections[di]
            t = self.tracks[tid]
            t.centroid = det["centroid"]
            t.bbox = det["bbox"]
            t.missed = 0
            t.age += 1
            t.visible = True
            t.history.append(det["centroid"])
            if len(t.history) > 120:
                t.history.pop(0)
            matched_dets.add(di)
            unmatched_track_ids.discard(tid)

        # New tracks for detections that matched nothing.
        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            t = Track(self.next_id, det["centroid"], det["bbox"])
            self.tracks[self.next_id] = t
            self.next_id += 1

        # Age out unmatched tracks; delete once past the grace window.
        deleted = []
        for tid in unmatched_track_ids:
            t = self.tracks[tid]
            t.missed += 1
            t.age += 1
            if t.missed > self.grace_frames:
                deleted.append(t)
        for t in deleted:
            del self.tracks[t.id]

        return self.tracks, deleted
