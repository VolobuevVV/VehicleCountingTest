
import sys
import traceback
import tensorflow as tf
from proc.bbox_tracker import BboxTracker
from proc.utils import *


class Detector:
    def __init__(self):
        self.detect_frame_step = 3
        self.anchors = [
            [0.57273, 0.677385],
            [1.87446, 2.06253],
            [3.33843, 5.47434],
            [7.88282, 3.52778],
            [9.77052, 9.16828],
        ]
        self.in_image_size = 224
        self.cell_size = 7
        self.boxes_per_cell = 5
        self.threshold = 0.85
        self.iou_threshold = 0.5
        self.input_node = "images:0"
        self.output_node = "Reshape:0"
        self.weight_file = 'proc/mobilenet_cars3.pb'
        self.car_count_left = 0
        self.car_count_right = 0
        self.bbox_trackers = BboxTracker()
        self.frames_counter = 0
        self.need_stream = True

    def load_graph(self):
        with tf.io.gfile.GFile(self.weight_file, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
        return graph

    def detect(self, img, sess, predictions_plch, input_plch):
        img_h, img_w, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inputs = img / 127.5 - 1
        inputs = np.reshape(inputs, (1, self.in_image_size, self.in_image_size, 3))

        netout = sess.run(predictions_plch, feed_dict={input_plch: inputs})[0]

        labels = preprocess_netout(netout,
                                   self.cell_size,
                                   self.anchors,
                                   self.threshold,
                                   scale_w=img_w,
                                   scale_h=img_h)

        labels = non_max_suppression_fast(labels, self.iou_threshold)
        return labels

    def queue_detect(self, input_queue, output_queue):
        print("INTERVAL")
        graph = self.load_graph()

        input_plch = graph.get_tensor_by_name("images:0")
        predictions_plch = graph.get_tensor_by_name("Reshape:0")

        with tf.compat.v1.Session(graph=graph) as sess:
            while True:
                try:
                    sys.stdout.flush()
                    val = input_queue.get()
                    line_divider, frame_time, frame = val[0], val[1], val[2]

                    labels = []
                    if self.frames_counter % self.detect_frame_step == 0 or self.frames_counter == -1:
                        labels = self.detect(frame, sess, predictions_plch, input_plch)
                        self.frames_counter = 0
                    else:
                        self.frames_counter += 1

                    bboxes, car_number_left, car_number_right = self.bbox_trackers.update(frame, labels, line_divider, 5, 1)
                    if car_number_left > 0:
                        self.car_count_left += car_number_left
                    if car_number_right > 0:
                        self.car_count_right += car_number_right
                    if self.need_stream:
                        frame = draw(bboxes, frame, line_divider, [(0, 255, 0)], self.car_count_left,
                                     self.car_count_right)
                        _, buf = cv2.imencode('.jpg', frame)
                        output_queue.put(buf.tobytes())

                except Exception as ex:
                    print(traceback.print_exc())
                finally:
                    pass
