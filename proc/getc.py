import cv2


# Функция для обработки кликов мыши
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:  # Проверяем, был ли клик левой кнопкой мыши
        # Получаем размеры кадра
        h, w, _ = frame.shape

        # Вычисляем относительные координаты
        relative_x = x / w
        relative_y = y / h

        # Отображаем координаты на изображении
        coords = f'X: {relative_x:.2f}, Y: {relative_y:.2f}'  # Форматируем до двух знаков после запятой
        print(coords)

        # Записываем координаты на изображение
        cv2.putText(frame, coords, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow('Frame', frame)  # Обновляем окно


# Открываем видеопоток (0 - это камера по умолчанию)
cap = cv2.VideoCapture(
    r"C:\Users\vladi\Downloads\vecteezy_image-of-traffic-on-the-road-passing-between-buildings-in_23272130.mov")

# Считываем один кадр
ret, frame = cap.read()

if ret:
    # Создаем окно для отображения кадра
    cv2.imshow('Frame', frame)

    # Устанавливаем обработчик событий мыши
    cv2.setMouseCallback('Frame', click_event)

    # Ожидаем нажатия клавиши 'q' для выхода
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
