import cv2
from picamera2 import Picamera2
import os

def stream_video():
    # Configurar Picamera2
    picam = Picamera2()
    picam.preview_configuration.main.size = (1280, 720)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    # Crear carpeta para almacenar las imágenes si no existe
    save_path = os.path.dirname(os.path.abspath(__file__))  # Carpeta del script actual
    img_count = 0  # Contador de imágenes

    while True:
        # Capturar frame
        frame = picam.capture_array()
        cv2.imshow("picam", frame)

        # Detectar teclas
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Salir con la tecla 'q'
            break
        elif key == ord('f'):  # Guardar imagen con la tecla 'f'
            img_name = os.path.join(save_path, f"foto_{img_count}.jpg")
            cv2.imwrite(img_name, frame)
            print(f"Foto guardada: {img_name}")
            img_count += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_video()
