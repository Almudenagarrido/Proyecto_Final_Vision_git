import cv2
import time
from picamera2 import Picamera2

expected_sequence = ["circle", "line", "square", "triangle"]

# Inicializar la memoria de los patrones detectados
detected_sequence = []
last_detection_time = 0  # Almacena el último tiempo de detección de forma
detection_delay = 1  # Intervalo en segundos entre detecciones
checking_sequence = False  # Bandera para indicar si se está comprobando la secuencia

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
            shapes.append(("line", contour, (255, 255, 255)))  # Blanco
        elif len(approx) == 3:
            shapes.append(("triangle", contour, (0, 255, 0)))  # Verde
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            if 0.9 <= w / float(h) <= 1.1:
                shapes.append(("square", contour, (255, 0, 0)))  # Azul
        elif len(approx) >= 8:
            shapes.append(("circle", contour, (0, 0, 255)))  # Rojo
    return shapes

def update_sequence(shapes):
    """Actualizar la secuencia detectada y verificar si es correcta."""
    global detected_sequence, checking_sequence, last_detection_time

    # Limitar detecciones rápidas consecutivas
    current_time = time.time()
    if current_time - last_detection_time < detection_delay:
        return f"Shapes detected: {len(detected_sequence)}/4"

    for shape, contour, color in shapes:
        if len(detected_sequence) < 4:
            detected_sequence.append(shape)
            last_detection_time = current_time  # Actualizar tiempo de detección
            print(f"Shape added: {shape}. Total shapes in sequence: {len(detected_sequence)}")

    if len(detected_sequence) == 4:
        checking_sequence = True  # Activar comprobación
        return "Checking Sequence..."
    return f"Shapes detected: {len(detected_sequence)}/4"


if __name__ == "__main__":
    
    # Configurar Picamera2
    picam = Picamera2()
    picam.preview_configuration.main.size = (1280, 720)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    print("Detecting shapes in real-time. Press 'q' to quit.")

    while True:
        # Capturar frame
        frame = picam.capture_array()

        # Detectar formas geométricas en el frame
        shapes = detect_shapes(frame)

        # Dibujar contornos y nombres en el frame
        for shape, contour, color in shapes:
            cv2.drawContours(frame, [contour], -1, color, 2)
            x, y, _, _ = cv2.boundingRect(contour)
            cv2.putText(frame, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if checking_sequence:
            # Comprobar la secuencia detectada
            if detected_sequence == expected_sequence:
                cv2.putText(frame, "Sequence Correct!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print("Correct sequence. Sequence Correct!")
                break
            else:
                cv2.putText(frame, "Access Denied! Try Again", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("Wrong sequence. Access Denied!")
            detected_sequence.clear()
            checking_sequence = False  # Resetear la comprobación
        else:
            # Actualizar y validar la secuencia
            result = update_sequence(shapes)
            cv2.putText(frame, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Mostrar el video en tiempo real
        cv2.imshow("Shape Detection", frame)

        # Salir con la tecla 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Salir con 'q'
            break

    picam.stop()
    cv2.destroyAllWindows()