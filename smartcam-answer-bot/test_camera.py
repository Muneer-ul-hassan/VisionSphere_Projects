from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()

frame_display = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    # Convert BGR to RGB for matplotlib
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    if frame_display is None:
        frame_display = ax.imshow(annotated_frame_rgb)
    else:
        frame_display.set_data(annotated_frame_rgb)

    plt.axis('off')
    plt.pause(0.001)  # short pause to refresh frame

    # Optional: break on 'q' key
    if plt.get_fignums() == []:  # window closed
        break

cap.release()
plt.ioff()
plt.close()
