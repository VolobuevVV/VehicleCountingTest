from multiprocessing import Process
from threading import Thread
import multiprocessing
from proc.camera_reader import CameraReader
import proc.detector_interval as det_interval
import proc.detector_area as det_area
from proc import DB as db
from server import serve

if __name__ == "__main__":

    video_path = 'test/forest_15sec.avi'
    m = multiprocessing.Manager()
    input_queue = m.Queue(maxsize=1000)
    output_queue = m.Queue(maxsize=1000)
    type_d = 1

    if type_d == 1:
        detector = det_area.Detector()
    else:
        detector = det_interval.Detector()

    db = db.DB()

    detector_process = Thread(target=detector.queue_detect, args=(input_queue, output_queue, ))
    detector_process.daemon = True
    detector_process.start()

    camera_reader = CameraReader(in_image_size=224,
                                 stream_url=video_path)

    camera_reader_process = Process(target=camera_reader.read, args=(input_queue,))
    camera_reader_process.daemon = True
    camera_reader_process.start()

    db_process = Process(target=db.run, args=(output_queue,))
    db_process.daemon = True
    db_process.start()

    server_process = Process(target=serve, args=())
    server_process.daemon = False
    server_process.start()
    #video_process = Process(target=create_video_from_queue, args=(output_queue, 'output_video.avi'))
    #video_process.daemon = True
    #video_process.start()


    while True:
        try:
            if not detector_process.is_alive():
                detector_process = Thread(target=detector.queue_detect, args=(input_queue, output_queue, ))
                detector_process.daemon = True
                detector_process.start()
            if not camera_reader_process.is_alive():
                camera_reader_process = Process(target=camera_reader.read, args=(input_queue, ))
                camera_reader_process.daemon = True
                camera_reader_process.start()
            if not db_process.is_alive():
                db_process = Process(target=db.run, args=(output_queue,))
                db_process.daemon = True
                db_process.start()
            if not server_process.is_alive():
                server_process = Process(target=serve, args=())
                server_process.daemon = False
                server_process.start()

        except KeyboardInterrupt:
            print("Программа была прервана пользователем")
            exit()

