import math
import logging
import uuid

import cv2
import numpy as np


def create_video_from_queue(output_queue, output_file, frame_rate=25, width=224, height=224):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height))

    while True:
        try:
            frame_bytes = output_queue.get(timeout=5)
            frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)

            if frame is None:
                print("Не удалось декодировать изображение.")
                continue

            if frame.shape[1] != width or frame.shape[0] != height:
                print(f"Неверный размер кадра: {frame.shape[1]}x{frame.shape[0]}. Ожидалось: {width}x{height}.")
                continue

            out.write(frame)
            cv2.imshow('Video Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Выход из программы.")
                break

        except Exception as e:
            print(f"Ошибка при получении кадра из очереди: {e}")


    out.release()  # Освобождаем ресурсы
    cv2.destroyAllWindows()  # Закрываем все окна OpenCV

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def iou(box1, box2):
    tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
         max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
    lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
         max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
    inter = 0 if tb < 0 or lr < 0 else tb * lr
    return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)


def draw(labels, img, line_divider, colors, count_left, count_right):
    class_idx = 0

    # Рисуем прямоугольники для каждого объекта
    for label in labels:
        x, y, w, h = np.int32(label[:4])
        cv2.rectangle(img, (x, y), (x + w, y + h), colors[class_idx], 1)


    # Рисуем линию разделителя
    x1, y1, x2, y2 = line_divider
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Цвет линии - синий (BGR)

    # Добавляем текст с значением count в верхней части изображения
    font = cv2.FONT_HERSHEY_SIMPLEX
    text1 = f'Count: {count_left + count_right}'
    #text2 = f'Count_right: {count_right}'
    text_position1 = (10, 30)  # Позиция текста (x, y)
    #text_position2 = (10, 60)  # Позиция текста (x, y)
    cv2.putText(img, text1, text_position1, font, 0.5, (255, 255, 255), 2)  # Белый цвет текста
    #cv2.putText(img, text2, text_position2, font, 0.5, (255, 255, 255), 2)  # Белый цвет текста

    return img


def preprocess_netout(data, cell_size, anchors, threshold, scale_w=1.0, scale_h=1.0, win_size_thresh=0.1):
    labels = []

    data[..., 2:4] = np.exp(data[..., 2:4]) / float(cell_size)
    data[..., 0:2] = sigmoid(data[..., 0:2])
    data[..., 4] = sigmoid(data[..., 4])

    for i in range(cell_size):
        for j in range(cell_size):
            for k in range(len(anchors)):
                #print data.shape
                class_prob = data[i, j, k, 5:] * data[i, j, k, 4]

                if class_prob > threshold:
                    w = int(data[i, j, k, 2] * anchors[k][0] * scale_w)
                    h = int(data[i, j, k, 3] * anchors[k][1] * scale_h)
                    x = int((j + data[i, j, k, 0]) / float(cell_size) * scale_w) - w / 2.
                    y = int((i + data[i, j, k, 1]) / float(cell_size) * scale_h) - h / 2.

                    if w / float(scale_w) > win_size_thresh and h / float(scale_h) > win_size_thresh:
                        labels.append(np.concatenate([[x, y, w, h], class_prob]))

    return np.array(labels)


"""
def non_max_suppression(classes, locations, threshold, iou_threshold):
    classes = np.transpose(classes)
    indxs = np.argsort(-classes, axis=1)

    for i in range(classes.shape[0]):
        classes[i] = classes[i][indxs[i]]

    for class_idx, class_vec in enumerate(classes):
        for roi_idx, roi_prob in enumerate(class_vec):
            if roi_prob < threshold:
                classes[class_idx][roi_idx] = 0

    for class_idx, class_vec in enumerate(classes):
        for roi_idx, roi_prob in enumerate(class_vec):

            if roi_prob == 0:
                continue
            roi = locations[indxs[class_idx][roi_idx]]

            for roi_ref_idx, roi_ref_prob in enumerate(class_vec):

                if roi_ref_prob == 0 or roi_ref_idx <= roi_idx:
                    continue

                roi_ref = locations[indxs[class_idx][roi_ref_idx]]

                if iou(roi, roi_ref) > iou_threshold:
                    classes[class_idx][roi_ref_idx] = 0

    return classes, indxs
"""


def non_max_suppression_fast(labels, iou_threshold):
    # if there are no boxes, return an empty list
    if len(labels) == 0:
        return []

    # Convert labels to a NumPy array if it is not already
    labels = np.array(labels)

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = labels[:, 0]
    y1 = labels[:, 1]
    x2 = labels[:, 0] + labels[:, 2]
    y2 = labels[:, 1] + labels[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater than the threshold
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > iou_threshold)[0])))

    return labels[pick]



def get_image_bbox_vals(image, bbox, bbox_offset=5):
    bbox = list(bbox)
    bbox[0] = int(bbox[0])
    bbox[1] = int(bbox[1])
    bbox[2] = int(bbox[2])
    bbox[3] = int(bbox[3])
    y_0, y_1 = min(bbox[1] - bbox_offset, 0), bbox[1] + bbox[3] + bbox_offset
    x_0, x_1 = min(bbox[0] - bbox_offset, 0), bbox[0] + bbox[2] + bbox_offset

    crop = image[y_0: y_1, x_0: x_1]

    mean = np.mean(crop)
    std = np.sum(crop - mean)

    return mean, std


def line_side(bbox1, line):
    if line is None:
        return -1
    else:
        lx1, ly1, lx2, ly2 = line
        bx1, by1, bx2, by2 = bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]
        cbx, cby = bx1 + bbox1[2] / 2., by1 + bbox1[3] / 2.
        s = (lx2 - lx1) * (cby - ly1) - (ly2 - ly1) * (cbx - lx1)
        s = -1 if (s > 0) else 1

    return s


def dist(bbox1, bbox2, line, frame, iou_thresh=0.8, max_return_val=1000):

    ls1 = None
    ls2 = None
    if line is not None:
        lx1, ly1, lx2, ly2 = line[0], line[1], line[2], line[3]
        ls1 = line_side(bbox1, (lx1, ly1, lx2, ly2))
        ls2 = line_side(bbox2, (lx1, ly1, lx2, ly2))

    if ls1 != ls2:
        return max_return_val
    else:
        bx1, by1, bx2, by2 = map(int, [bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]])
        cbx, cby = bx1 + bbox1[2] / 2., by1 + bbox1[3] / 2.
        btx1, bty1, btx2, bty2 = map(int, [bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]])
        cbtx, cbty = btx1 + bbox2[2] / 2., bty1 + bbox2[3] / 2.

        if iou([bx1, by1, bx2, by2], [btx1, bty1, btx2, bty2]) >= iou_thresh:
            # not create new tracker
            return 0

        m1, std1 = get_image_bbox_vals(frame, list(bbox1))
        m2, std2 = get_image_bbox_vals(frame, list(bbox2))

        mean_diff = abs(m2 - m1)
        mean_diff = 0 if np.isnan(mean_diff) else mean_diff
        len_bbox_vec = math.sqrt((cbx - cbtx) ** 2 + (cby - cbty) ** 2)

        if line is not None:
            vec1 = (cbtx - cbx, cbty - cby)
            vec2 = (lx2 - lx1, ly2 - ly1)

            cosvec = 1 - abs(np.sum(np.multiply(vec1, vec2)) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            return len_bbox_vec + len_bbox_vec * (cosvec * 3) ** 3 + mean_diff
        else:
            return len_bbox_vec + mean_diff


def frames_std_diff(frame1, frame2):
    diff32 = frame1 - frame2
    norm32 = np.sqrt(np.sum(np.float32(diff32) ** 2, 2)) / np.sqrt(255 ** 2 + 255 ** 2 + 255 ** 2)
    dist = np.uint8(norm32 * 255)

    mod = cv2.GaussianBlur(dist, (15, 15), 0)
    _, thresh = cv2.threshold(mod, 100, 255, 0)
    _, stDev = cv2.meanStdDev(mod)

    #print stDev

    return stDev


