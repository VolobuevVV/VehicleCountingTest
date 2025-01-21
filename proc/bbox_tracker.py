import proc.utils as utils

import cv2
import uuid

class BboxTracker:
    def __init__(self):
        self.tracker_type = cv2.TrackerMOSSE_create()
        self.trackers = {}
        self.bbox_dist_thresh = 150


    def add_tracker(self, frame, box, id=None):
        tracker = self.tracker_type
        tracker.init(frame, tuple(box))
        if id is None:
            id = str(uuid.uuid4())
            status = False
        else:
            status = self.trackers[id][3]
        self.trackers[id] = [tracker, box, 0, status]

    def update_tracker(self, frame, key, tracker):
        retl, bbox = tracker.update(frame)
        if retl:
            self.trackers[key] = [tracker, bbox, self.trackers[key][2], self.trackers[key][3]]

    def update(self, frame, rects, line_divider, frames_release, frames_decision):
        h, w = frame.shape[:2]
        new_car_number_left = 0
        new_car_number_right = 0

        for key, tr in self.trackers.items():
            self.update_tracker(frame, key, tr[0])

        new_net_bboxes = [list(map(int, bbox[:4])) for bbox in rects]

        if new_net_bboxes:
            if self.trackers:
                d_arr = []
                for new_box in new_net_bboxes:
                    for key, tr in self.trackers.items():
                        d_arr.append((key, utils.dist(tr[1], new_box, line_divider, frame), tr[1]))

                for new_box in new_net_bboxes:
                    dist_arr = sorted(d_arr, key=lambda x: x[1])
                    if dist_arr and dist_arr[0][1] > self.bbox_dist_thresh:
                        self.add_tracker(frame, new_box)
                    else:
                        self.add_tracker(frame, new_box, dist_arr[0][0])

            else:
                for bbox in new_net_bboxes:
                    self.add_tracker(frame, bbox)

        bboxes = []
        keys_to_remove = []

        for key in list(self.trackers.keys()):
            tr = self.trackers[key]
            bbox = tr[1]
            frames_number = tr[2]
            cx, cy = bbox[0] + bbox[2] / 2., bbox[1] + bbox[3] / 2.
            b = 7
            if cy >= h - b or cy <= b or cx >= w - b or cy <= b or frames_number > frames_release:
                keys_to_remove.append(key)
            else:
                bboxes.append(bbox)
                if frames_number == frames_decision and not tr[3]:
                    side = utils.line_side(bbox, line_divider)
                    if side < 0:
                        new_car_number_left += 1
                    else:
                        new_car_number_right += 1
                    tr[3] = True
                tr[2] += 1

        for key in keys_to_remove:
            self.trackers.pop(key)

        return bboxes, new_car_number_left, (new_car_number_right if line_divider is not None else 0)
