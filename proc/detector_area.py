import sys
import traceback
import tensorflow as tf
from proc.bbox_tracker import BboxTracker
from proc.utils import *


class Detector:
    def __init__(self):
        self.detect_frame_step = 3
        self.in_image_size = 224
        self.threshold = 0.3
        self.iou_threshold = 0.5
        self.weight_file = 'proc/frozen_inference_graph_old.pb'

        self.bbox_trackers = BboxTracker()
        self.frames_counter = 0
        self.car_detected = False
        self.car_count_left = 0
        self.car_count_right = 0
        self.need_stream = False
        self.polygon_points = np.array([
            [0.23 * 224, 0.25 * 224],  # Левый верхний угол
            [0.43 * 224, 0.15 * 224],  # Правый верхний угол
            [1 * 224, 0.12 * 224],  # Правый нижний угол
            [0.7 * 224, 1 * 224]
        ], dtype=np.int32)

        self.need_classes = [3, 4, 6, 7, 8, 9]

    def load_graph(self):
        with tf.io.gfile.GFile(self.weight_file, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
        return graph


    def detect(self, img, sess):
        img_h, img_w, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        inputs = np.reshape(img, (1, self.in_image_size, self.in_image_size, 3))

        out = sess.run(
            [
                sess.graph.get_tensor_by_name('num_detections:0'),
                sess.graph.get_tensor_by_name('detection_scores:0'),
                sess.graph.get_tensor_by_name('detection_boxes:0'),
                sess.graph.get_tensor_by_name('detection_classes:0')
            ],
            feed_dict={'image_tensor:0': inputs}
        )

        num_detections = int(out[0][0])
        boxes = out[2][0][:num_detections]
        scores = out[1][0][:num_detections]
        class_ids = out[3][0][:num_detections]

        valid_detections = scores > self.threshold
        valid_class_ids = np.array([cid in self.need_classes for cid in class_ids], dtype=bool)
        valid_combined = valid_detections & valid_class_ids

        filtered_boxes = boxes[valid_combined]
        filtered_scores = scores[valid_combined]

        x_coords = filtered_boxes[:, 1] * img_w
        y_coords = filtered_boxes[:, 0] * img_h
        widths = (filtered_boxes[:, 3] * img_w) - x_coords
        heights = (filtered_boxes[:, 2] * img_h) - y_coords

        labels = np.column_stack((x_coords, y_coords, widths, heights, filtered_scores))

        labels = non_max_suppression_fast(labels.tolist(), self.iou_threshold)

        return labels

    def queue_detect(self, input_queue, output_queue):
        val = input_queue.get()
        line_divider, frame_time, frame = val[0], val[1], val[2]
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [self.polygon_points], (255, 255, 255))
        graph = self.load_graph()
        with tf.compat.v1.Session(graph=graph) as sess:
            while True:
                try:
                    sys.stdout.flush()
                    val = input_queue.get(timeout=1)
                    line_divider, frame_time, frame = val[0], val[1], val[2]
                    #print(frame_time)
                    frame = cv2.bitwise_and(frame, mask)

                    labels = []
                    if self.frames_counter % self.detect_frame_step == 0 or self.frames_counter == -1:
                        labels = self.detect(frame, sess)
                        self.frames_counter = 0
                    else:
                        self.frames_counter += 1

                    bboxes, car_number_left, car_number_right = self.bbox_trackers.update(frame, labels, line_divider,  15, 2)
                    if car_number_left > 0 or car_number_right > 0:
                        if car_number_left > 0:
                            self.car_count_left += car_number_left
                        if car_number_right > 0:
                            self.car_count_right += car_number_right
                        output_queue.put((self.car_count_left, self.car_count_right, frame_time))

                    #if self.need_stream:
                        #frame = draw(bboxes, frame, line_divider, [(0, 255, 0)], self.car_count_left, self.car_count_right)
                        #_, buf = cv2.imencode('.jpg', frame)
                        #output_queue.put(buf.tobytes())


                except Exception as ex:
                    print(str(ex))
                finally:
                    pass

