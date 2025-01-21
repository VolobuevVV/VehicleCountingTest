import cv2
import numpy as np


class ColorExtractor:
    def __init__(self):
        self.rectangles = []
        self.drawing = False
        self.current_rectangle = []

    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_rectangle = [(x, y), (x, y)]  # Инициализируем с двумя точками
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_rectangle[1] = (x, y)  # Обновляем вторую точку
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.rectangles.append((
                min(self.current_rectangle[0][0], self.current_rectangle[1][0]),
                min(self.current_rectangle[0][1], self.current_rectangle[1][1]),
                abs(self.current_rectangle[1][0] - self.current_rectangle[0][0]),
                abs(self.current_rectangle[1][1] - self.current_rectangle[0][1])
            ))
            self.current_rectangle = []

    def extract_color(self, frame, bbox):
        # Извлечение среднего цвета в bounding box
        x1, y1, w, h = bbox
        roi = frame[y1:y1 + h, x1:x1 + w]
        mean_color = cv2.mean(roi)[:3]  # Получаем средний цвет (B, G, R)
        return mean_color

    def run(self, image_path):
        # Загружаем изображение
        original_image = cv2.imread(image_path)
        # Уменьшаем изображение до 224x224
        image = cv2.resize(original_image, (224, 224))

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.draw_rectangle)

        while True:
            img_copy = image.copy()
            for rect in self.rectangles:
                cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)

            if self.drawing and len(self.current_rectangle) == 2:
                cv2.rectangle(img_copy, self.current_rectangle[0], self.current_rectangle[1], (255, 0, 0), 2)

            cv2.imshow("Image", img_copy)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # Пробел
                # Изменяем размеры прямоугольников обратно к оригинальному изображению
                resized_rectangles = [
                    (
                        int(rect[0] * original_image.shape[1] / 224),
                        int(rect[1] * original_image.shape[0] / 224),
                        int(rect[2] * original_image.shape[1] / 224),
                        int(rect[3] * original_image.shape[0] / 224)
                    ) for rect in self.rectangles
                ]

                colors = [self.extract_color(original_image, rect) for rect in resized_rectangles]
                print("Средние цвета:", colors)
                return colors

            elif key == 27:  # Esc
                break

        cv2.destroyAllWindows()


def color_distance(color1, color2):
    return np.linalg.norm(np.array(color1) - np.array(color2))

if __name__ == "__main__":
    extractor = ColorExtractor()
    colors = extractor.run(r"C:\Users\vladi\Downloads\cars.jpg")  # Замените на путь к вашему изображению
    if len(colors) >= 2:  # Убедимся, что извлечено достаточно цветов
        color1 = colors[0]
        color2 = colors[1]
        print(f"Цвет 1: {color1}, Цвет 2: {color2}")
        print("Расстояние между цветами:", color_distance(color1, color2))
    else:
        print("Недостаточно цветов для вычисления расстояния.")
