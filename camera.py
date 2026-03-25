from picamera2 import Picamera2, Preview
import cv2
import threading
import numpy as np

def setup_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (2592, 1944)})
    config["controls"]["ExposureTime"] = 20000  # Valor inicial
    config["controls"]["AnalogueGain"] = 1.0
    picam2.configure(config)
    return picam2

def adjust_exposure(picam2, img):
    mean_intensity = np.mean(img)
    current_exposure = picam2.camera_controls["ExposureTime"].value
    if mean_intensity > 180:  # Reduzir se muito brilhante
        new_exposure = max(5000, current_exposure - 1000)
        picam2.set_controls({"ExposureTime": new_exposure})
    elif mean_intensity < 70:  # Aumentar se muito escuro
        new_exposure = min(30000, current_exposure + 1000)
        picam2.set_controls({"ExposureTime": new_exposure})
    return picam2.capture_array()

def start_preview(picam2):
    def preview_loop():
        cv2.namedWindow("PalmTech Preview", cv2.WINDOW_NORMAL)
        while True:
            frame = picam2.capture_array()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("PalmTech Preview", gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    thread = threading.Thread(target=preview_loop, daemon=True)
    thread.start()
    picam2.start()

def stop_preview(picam2):
    picam2.stop_preview()
    picam2.close()
