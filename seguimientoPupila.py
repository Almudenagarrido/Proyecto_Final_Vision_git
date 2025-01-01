import cv2
import time
import numpy as np
from picamera2 import Picamera2

# Definir la secuencia esperada para la contraseña
expected_sequence = ["Circulo", "Linea", "Cuadrado", "Triangulo"]

# Inicializar la memoria de los patrones detectados
detected_sequence = []
last_detection_time = 0  # Almacena el último tiempo de detección de forma
detection_delay = 1  # Intervalo en segundos entre detecciones
checking_sequence = False  # Bandera para indicar si se está comprobando la secuencia
sequence_correct = False  # Bandera para indicar si la secuencia es correcta
seguir_pintando = True
initialize_kalman = False
detected_eyes = None
frame_guardado = None
buscar_pupilas = False

def detect_shapes(frame):
    """Detectar y clasificar las formas geométricas más cercanas al centro."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar contornos por tamaño y posición
    min_area = 1000  # Área mínima inicial
    frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)  # Centro del frame
    max_distance = frame.shape[0] // 4  # Distancia máxima desde el centro

    shapes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:  # Filtrar formas pequeñas
            continue

        # Calcular el centroide del contorno
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

        # Filtrar formas lejanas del centro
        distance = ((cx - frame_center[0])**2 + (cy - frame_center[1])**2)**0.5
        if distance > max_distance:
            continue

        # Clasificar la forma
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 2:
            shapes.append(("Linea", contour, (255, 255, 255)))  # Blanco
        elif len(approx) == 3:
            shapes.append(("Triangulo", contour, (0, 255, 0)))  # Verde
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            if 0.9 <= w / float(h) <= 1.1:
                shapes.append(("Cuadrado", contour, (255, 0, 0)))  # Azul
        elif len(approx) >= 8:
            shapes.append(("Circulo", contour, (0, 0, 255)))  # Rojo
    return shapes

def update_sequence(shapes):
    """Actualizar la secuencia detectada y verificar si es correcta."""
    global detected_sequence, checking_sequence, last_detection_time

    # Limitar detecciones rápidas consecutivas
    current_time = time.time()
    if current_time - last_detection_time < detection_delay:
        return f"Figuras detectadas: {len(detected_sequence)}/4"

    for shape, contour, color in shapes:
        if len(detected_sequence) < 4:
            detected_sequence.append(shape)
            last_detection_time = current_time  # Actualizar tiempo de detección
            print(f"Figura añadida: {shape}. Total de figuras en secuencia: {len(detected_sequence)}")

    if len(detected_sequence) == 4:
        checking_sequence = True  # Activar comprobación
        return "Comprobando secuencia..."
    return f"Figuras detectadas: {len(detected_sequence)}/4"

def detect_eyes(frame):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detectar círculos usando la Transformada de Hough
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=50
    )

    # Si se detectan círculos
    if circles is not None:
        circles = np.uint16(np.around(circles))
        eyes = []

        # Filtrar los dos círculos más grandes (posiblemente los ojos)
        sorted_circles = sorted(circles[0, :], key=lambda c: c[2], reverse=True)
        for i in range(min(2, len(sorted_circles))):
            x, y, radius = sorted_circles[i]
            eyes.append((x, y, radius))
            # Dibujar los círculos en la imagen
            cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)  # Contorno
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)  # Centro

        return eyes, frame

    else:
        return None, None

def initialize_kalman_filter():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    return kf

# Inicializa Kalman para ambas pupilas
kalman_filters = [initialize_kalman_filter(), initialize_kalman_filter()]

# Inicializar la captura de video
picam = Picamera2()
picam.preview_configuration.main.size = (1280, 720)
picam.preview_configuration.main.format = "RGB888"
picam.preview_configuration.align()
picam.configure("preview")
picam.start()
print("Detectando formas en tiempo real. Presione 'q' para salir")

# Configurar el grabador de video
frame_width = 1280
frame_height = 720  # Alto del frame
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codificador de video
out = cv2.VideoWriter('video_seguimiento_raspi.mp4', fourcc, 20.0, (frame_width, frame_height))  # Nombre y configuración del video

while True:
    frame = picam.capture_array()

    # Detectar formas geométricas en el frame
    shapes = detect_shapes(frame)

    # Dibujar contornos y nombres en el frame
    if seguir_pintando:
        for shape, contour, color in shapes:
            cv2.drawContours(frame, [contour], -1, color, 2)
            x, y, _, _ = cv2.boundingRect(contour)
            cv2.putText(frame, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if checking_sequence:
        # Comprobar la secuencia detectada
        if detected_sequence == expected_sequence:
            cv2.putText(frame, "Secuencia correcta!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print("Secuencia correcta!")
            sequence_correct = True  # Marcar como correcta la secuencia
            inicio = time.time()
            checking_sequence = False  # Desactivar comprobación de secuencia
            seguir_pintando = False
        else:
            cv2.putText(frame, "Acceso denegado! Vuelve a intentarlo.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print("Acceso denegado! Vuelve a intentarlo.")
            checking_sequence = False
        detected_sequence.clear()
        
    elif sequence_correct:
        
        # Fase de seguimiento de pupila
        if time.time() - inicio >= 5:
            cv2.putText(frame, "Siguiendo pupilas...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            if buscar_pupilas:
                detected_eyes, frame_ojos = detect_eyes(frame_guardado.copy())  # (x, y, radio)
            
                if detected_eyes is None:
                    print("No se han detectado pupilas en el frame guardado.")
                    frame_guardado = frame.copy()
                    
                else:
                    detected_eyes = sorted(detected_eyes, key=lambda e: e[0])  # Ordena por la coordenada x

                    cv2.putText(frame_ojos, "Pupilas detectadas!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Eyes Detected", frame_ojos)
                    while True:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                    cv2.destroyAllWindows()
                    print("Pupilas detectadas, guardadas en 'detected_eyes'")
                    buscar_pupilas = False
             
            # Una vez tenemos la posición inicial (detected_eyes no es None)
            if detected_eyes is not None and not initialize_kalman:
                for i, (px, py, r) in enumerate(detected_eyes[:2]):
                    kalman_filters[i].statePost = np.array([[px], [py], [0], [0]], np.float32)
                    kalman_filters[i].errorCovPost = np.eye(4, dtype=np.float32)
                initialize_kalman = True
                
            if initialize_kalman:
                new_eyes, frame_ojos = detect_eyes(frame)
            
                if new_eyes is not None:
                    new_eyes = sorted(new_eyes, key=lambda e: e[0])  # Ordena por la coordenada x

                    # Mezclamos la detección con Kalman
                    for i, kf in enumerate(kalman_filters):
                        if i < len(new_eyes):
                            measured_x, measured_y, measured_r = new_eyes[i]
                            
                            # Predicción
                            prediction = kf.predict()
                            pred_x, pred_y = int(prediction[0]), int(prediction[1])

                            # Corrección
                            measurement = np.array([[np.float32(measured_x)], [np.float32(measured_y)]])
                            kf.correct(measurement)

                            # Dibujar la predicción
                            cv2.circle(frame, (pred_x, pred_y),5, (0, 0, 255), -1)  # Rojo
                            cv2.circle(frame, (measured_x, measured_y),5, (0, 255, 0), -1)  # Verde
                            cv2.rectangle(frame, (int(measured_x - measured_r), int(measured_y - measured_r)),
                                        (int(measured_x + measured_r), int(measured_y + measured_r)), (255, 0, 0), 2)
                            
                        else:
                            # No tenemos más mediciones
                            prediction = kf.predict()
                            pred_x, pred_y = int(prediction[0]), int(prediction[1])
                            cv2.circle(frame, (pred_x, pred_y), 5, (0, 0, 255), -1)
                
                else:
                    # Si no detectamos nada, al menos hacemos la predicción
                    for kf in kalman_filters:
                        prediction = kf.predict()
                        pred_x, pred_y = int(prediction[0]), int(prediction[1])
                        cv2.circle(frame, (pred_x, pred_y), 5, (0, 0, 255), -1)
            
        elif time.time() - inicio < 5:
            frame_guardado = frame.copy()
            cv2.putText(frame, "Detectando pupilas...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            buscar_pupilas = True

    else:
        # Actualizar y validar la secuencia
        result = update_sequence(shapes)
        cv2.putText(frame, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Guardar el frame en el archivo de video
    out.write(frame)

    # Mostrar el video en tiempo real
    cv2.imshow("Shape Detection", frame)

    # Salir con la tecla 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Salir con 'q'
        break

# Liberar la captura y cerrar el archivo de video
out.release()
cv2.destroyAllWindows()
