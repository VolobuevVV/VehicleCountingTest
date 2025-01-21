import cv2
import sys
import time
import traceback
from proc.utils import *



class CameraReader(object):

    def __init__(self, in_image_size, stream_url):
        self.stream_url = stream_url
        self.frames_base_thresh = 40
        self.retrieve_frame = 2
        self.line_divider = (0.57, 0.92, 1,  0.3)
        self.in_image_size = in_image_size

    def read_cap(self):
        if 'rtsp' in self.stream_url:
            conn_str = "rtspsrc latency=1000 retry=100 timeout=100 " \
                       "location=%s " \
                       "! rtph264depay " \
                       "! h264parse " \
                       "! avdec_h264 " \
                       "! videoconvert " \
                       "! appsink max-buffers=10 drop=true max-lateness=500000000 sync=false" % self.stream_url
            cap = cv2.VideoCapture(conn_str, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
        return cap

    def read(self, input_queue):
        cap = self.read_cap()

        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        previous_frame = None
        ret = None
        c = 0
        line_divider = [int(ld * self.in_image_size) for ld in self.line_divider]
        print(line_divider)

        polygon_points = np.array([
            [0.23 * original_width, 0.25 * original_height],  # Левый верхний угол
            [0.43 * original_width, 0.15 * original_height],  # Правый верхний угол
            [1 * original_width, 0.12 * original_height],  # Правый нижний угол
            [0.7 * original_width, 1 * original_height]
        ], dtype=np.int32)

        while True:
            try:
                ret = cap.grab()
                if c % self.retrieve_frame == 0 or input_queue.qsize() == 0:
                    ret, frame = cap.retrieve()

                    if ret and frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                        if previous_frame is not None and input_queue.qsize() > 0:
                            if frames_std_diff(frame, previous_frame) > (self.frames_base_thresh + (5 ** (1 + input_queue.qsize() / 1000.))):
                                #mask = np.zeros_like(frame)
                                #cv2.fillPoly(mask, [polygon_points], (255, 255, 255))
                                #frame = cv2.bitwise_and(frame, mask)
                                input_queue.put((line_divider, int(time.time()), cv2.resize(frame, (self.in_image_size, self.in_image_size))))
                        else:
                            #mask = np.zeros_like(frame)
                            #cv2.fillPoly(mask, [polygon_points], (255, 255, 255))
                            #frame = cv2.bitwise_and(frame, mask)
                            input_queue.put((line_divider, int(time.time()), cv2.resize(frame, (self.in_image_size, self.in_image_size))))
                        previous_frame = frame
                        print("Input queue size: {}".format(input_queue.qsize()))
                    else:
                        print("Stream/Frame can't open. End of stream.")
                        cap = self.read_cap()
                        c = 0
                else:
                    if not ret:
                        print("Stream/Frame can't open. End of stream.")
                        cap = self.read_cap()
                        c = 0
                c += 1


            except Exception as ex:
                if ret is False:
                    print(traceback.format_exc())
                    print('Stream error:', str(ex))
                    sys.stdout.flush()
                else:
                    print(traceback.format_exc())
                    print('Inference error:', str(ex))
                    sys.stdout.flush()
            finally:
                pass
