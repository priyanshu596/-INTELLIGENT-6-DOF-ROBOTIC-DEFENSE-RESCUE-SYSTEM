import cv2
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from ultralytics import YOLO
import time

# --- Connect to CoppeliaSim ---
client = RemoteAPIClient()
sim = client.require('sim')
vision_sensor_handle = sim.getObject('/NiryoOne/visionSensor')

# --- Load pretrained YOLO drone model ---
yolo_model = YOLO("best.pt")  # replace with your downloaded weights

while True:
    # --- Capture camera feed ---
    img, resX, resY = sim.getVisionSensorCharImage(vision_sensor_handle)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
    img = cv2.flip(img, 0)

    # --- Run YOLO detection ---
    results = yolo_model(img, conf=0.1)  # lower confidence for simulation
    boxes = results[0].boxes

    # --- If YOLO detects drone ---
    if boxes and len(boxes) > 0:
        # Take first detected box
        x1, y1, x2, y2 = boxes[0].xyxy[0].tolist()
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cx, cy = (x1+x2)/2, (y1+y2)/2
    else:
        # --- Fallback: color detection for simulation (red sphere) ---
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cx, cy = x + w/2, y + h/2

            
        else:
            cx, cy = None, None



    # --- Draw center point ---
    if cx is not None and cy is not None:
        cv2.circle(img, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        # Observation for RL: dx, dy from image center
        dx = (cx - resX/2) / (resX/2)
        dy = (cy - resY/2) / (resY/2)
        # For now just print
        distance = ((cx - resX/2)**2 + (cy - resY/2)**2) ** 0.5
        normalized_distance = (dx**2 + dy**2) ** 0.5


        print(f"cx: {cx:.2f}, cy: {cy:.2f}")
        print(f"distance: {normalized_distance}")

    cv2.imshow("Drone Tracking", img)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cv2.destroyAllWindows()
