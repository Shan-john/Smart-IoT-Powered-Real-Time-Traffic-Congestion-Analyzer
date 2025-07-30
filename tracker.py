import math
import time

class SimpleTracker:
    def __init__(self, distance_threshold=40, stuck_seconds=5):
        self.tracked = {}  # id -> {'pos': (x, y), 'last_seen': time, 'stuck_since': time or None}
        self.next_id = 0
        self.threshold = distance_threshold
        self.stuck_seconds = stuck_seconds

    def update(self, detections):
        updated = []
        now = time.time()

        for x, y in detections:
            matched_id = None
            for obj_id, data in self.tracked.items():
                prev_x, prev_y = data['pos']
                dist = math.hypot(x - prev_x, y - prev_y)
                if dist < self.threshold:
                    matched_id = obj_id
                    break

            if matched_id is not None:
                data = self.tracked[matched_id]
                data['last_seen'] = now
                if math.hypot(x - data['pos'][0], y - data['pos'][1]) < 10:
                    if data['stuck_since'] is None:
                        data['stuck_since'] = now
                else:
                    data['stuck_since'] = None
                data['pos'] = (x, y)
                updated.append((matched_id, (x, y)))
            else:
                self.tracked[self.next_id] = {
                    'pos': (x, y),
                    'last_seen': now,
                    'stuck_since': None
                }
                updated.append((self.next_id, (x, y)))
                self.next_id += 1

        # Remove stale
        to_remove = [i for i, d in self.tracked.items() if now - d['last_seen'] > 5]
        for i in to_remove:
            del self.tracked[i]

        return updated

    def get_stuck_vehicles(self):
        now = time.time()
        return [(i, d['pos']) for i, d in self.tracked.items() if d['stuck_since'] and now - d['stuck_since'] >= self.stuck_seconds]
