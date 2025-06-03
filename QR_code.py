import cv2
import numpy as np

# ===================== Настройки =====================
VIDEO_PATH = r'D:\Rabota1\pythonProject\IMG_3914_ok.mp4'
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720

# Параметры камеры
focal_length = 1000
center = (VIDEO_WIDTH // 2, VIDEO_HEIGHT // 2)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype=np.float64)
dist_coeffs = np.zeros(4)

# Параметры Optical Flow
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

# ===================== Инициализация =====================
qr_map = {}
prev_gray = None
camera_pose = np.eye(4)


def detect_qr_codes(frame):
    detector = cv2.QRCodeDetector()
    retval, decoded_info, points, _ = detector.detectAndDecodeMulti(frame)

    detections = []
    if points is not None:
        for i, pts in enumerate(points):
            if decoded_info and i < len(decoded_info):
                pts = pts.astype(np.float32)
                detections.append({
                    'data': decoded_info[i],
                    'points': pts,
                    'center': np.mean(pts, axis=0)
                })
    return detections


def process_frame(frame):
    global prev_gray, qr_map

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detect_qr_codes(frame)

    for det in detections:
        data = det['data']
        if data:
            if data not in qr_map:
                qr_map[data] = {'points': det['points'], 'detected': True}
            else:
                qr_map[data].update({
                    'points': det['points'],
                    'detected': True
                })

    for data, qr in qr_map.items():
        if qr['detected']:
            pts = qr['points'].astype(int)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            center = tuple(map(int, qr['points'].mean(axis=0)))
            cv2.putText(
                frame,
                data,
                (center[0], center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )
            qr['detected'] = False

    cv2.putText(
        frame,
        f"Found QR: {sum(1 for q in qr_map.values() if q['detected'])}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    prev_gray = gray
    return frame


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Ошибка открытия видео: {VIDEO_PATH}")
        print("Проверьте:")
        print("1. Существует ли файл")
        print("2. Поддерживается ли формат")
        print("3. Попробуйте другой видеофайл")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

    for _ in range(5):
        cap.read()  # Пропускаем первые кадры

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if np.mean(frame) < 10:
            continue

        frame = process_frame(frame)
        cv2.imshow('QR Detection', frame)

        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
