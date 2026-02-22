# Object tracking means following an object in a video
# and giving it a unique ID that stays the same
# while the object appears in the video.


from ultralytics import YOLO
import cv2

# -----------------------
# Load YOLOv8 Model
# -----------------------
model = YOLO('yolov8n.pt')

# -----------------------
# Load Video
# -----------------------A
videopath = 'Video/video3.mp4'
cap = cv2.VideoCapture(videopath)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# -----------------------
# Process Video Frames
# -----------------------
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    print(frame_counter)

    frame = cv2.resize(frame, (1080, 720))

    # persist=True → 
    #   remembers object IDs from previous frames
    #   helps keep the same ID for the same object
    results = model.track(frame, persist=True)

    # Plot results
    frame = results[0].plot()

    # Show output
    cv2.imshow('Tracking', frame)

    frame_counter = frame_counter + 1

    # Exit with 's'
    if cv2.waitKey(0) & 0xFF == ord('s'):
        break

# -----------------------
# Release Resources
# -----------------------
cap.release()
cv2.destroyAllWindows()
