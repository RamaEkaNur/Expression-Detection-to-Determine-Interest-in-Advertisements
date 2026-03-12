import cv2
import time
from datetime import datetime
from ultralytics import YOLO

# Load the model
model_path = 'C:/Users/Rama Eka/Documents/Tugas/semester 7/comvis/test/skripsi/PrimerSekunder5_256.pt'
model = YOLO(model_path)

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variables for FPS calculation
prev_time = 0

# Callback function for trackbars (does nothing but required for createTrackbar)
def nothing(x):
    pass

# Create a window and trackbars for thresholds
cv2.namedWindow("Webcam")
cv2.createTrackbar("CONFIDENCE_THRESHOLD", "Webcam", 30, 100, nothing)  # Initial value 30, max 100
cv2.createTrackbar("NMS_THRESHOLD", "Webcam", 0, 100, nothing)          # Initial value 0, max 100

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video")
            break

        # Read the current slider positions and convert to appropriate scales
        CONFIDENCE_THRESHOLD = cv2.getTrackbarPos("CONFIDENCE_THRESHOLD", "Webcam") / 100
        NMS_THRESHOLD = cv2.getTrackbarPos("NMS_THRESHOLD", "Webcam") / 100

        # Start timer for computation time
        start_time = time.time()

        # Run inference on the frame
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, iou=NMS_THRESHOLD, stream=True)

        # Loop through results to process detections
        for result in results:
            boxes = result.boxes  # Access boxes in the result

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = box.conf[0]  # Confidence score
                cls = int(box.cls[0])  # Class index
                original_label = model.names[cls]

                # Draw a bounding box around the detected object
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

                # Create a label with confidence score
                label_text = f'{original_label} {conf:.2f}'

                # Draw a background rectangle for the label
                (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (255, 255, 255), thickness=cv2.FILLED)

                # Draw the label text
                cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Display FPS on the frame
        fps_text = f'FPS: {fps:.2f}'
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the frame with bounding boxes
        cv2.imshow("Webcam", frame)

        # Exit the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()






# import cv2
# import time
# import csv
# from datetime import datetime
# from ultralytics import YOLO

# # Load the model
# model_path = 'C:/Users/Rama Eka/Documents/Tugas/semester 7/comvis/test/skripsi/FER2013.pt'
# model = YOLO(model_path)

# # Initialize the webcam
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# # Set fixed confidence and NMS thresholds
# CONFIDENCE_THRESHOLD = 0.30  # 30%
# NMS_THRESHOLD = 0.0          # 0%

# # Variables for FPS calculation
# prev_time = 0

# # Generate a unique filename using real-time date and time
# realtime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# output_file = f"classification_results_{realtime}.csv"

# # Prepare to log output to CSV
# output_data = []

# try:
#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to capture video")
#             break

#         # Start timer for computation time
#         start_time = time.time()

#         # Run inference on the frame
#         results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, iou=NMS_THRESHOLD, stream=True)

#         # Loop through results to process detections
#         for result in results:
#             boxes = result.boxes  # Access boxes in the result

#             for box in boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
#                 conf = box.conf[0]  # Confidence score
#                 cls = int(box.cls[0])  # Class index
#                 original_label = model.names[cls]

#                 # Draw a bounding box around the detected object
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

#                 # Create a label with confidence score
#                 label_text = f'{original_label} {conf:.2f}'

#                 # Draw a background rectangle for the label
#                 (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
#                 cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (255, 255, 255), thickness=cv2.FILLED)

#                 # Draw the label text
#                 cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

#                 # Save output data to the list
#                 computation_time_ms = (time.time() - start_time) * 1000  # Convert to milliseconds
#                 current_time = datetime.now().strftime('%H:%M:%S')  # Current time
#                 output_data.append([current_time, f'{conf:.2f}', original_label, computation_time_ms])

#         # Calculate FPS
#         curr_time = time.time()
#         fps = 1 / (curr_time - prev_time)
#         prev_time = curr_time

#         # Display FPS on the frame
#         fps_text = f'FPS: {fps:.2f}'
#         cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#         # Display the frame with bounding boxes
#         cv2.imshow('Webcam', frame)

#         # Exit the loop on 'q' key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# finally:
#     # Release the webcam
#     cap.release()
#     cv2.destroyAllWindows()

#     # Write output data to CSV
#     with open(output_file, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         # Write header row
#         writer.writerow(["Time", "Confidence", "Classification", "Computation Time (ms)"])
#         # Write all rows
#         writer.writerows(output_data)

#     print(f"Results saved to {output_file}")

