import cv2
from ultralytics import YOLO
import pyzed.sl as sl

# Load YOLOv8 model
model = YOLO('collision_warning_yolov8.pt')

# Initialize ZED Camera
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.coordinate_units = sl.UNIT.METER

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("ZED Camera failed to initialize")
    exit()

runtime_params = sl.RuntimeParameters()
image = sl.Mat()

while True:
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)  # Get the left image
        frame = image.get_data()  # Convert to numpy array (BGRA format)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert to BGR for OpenCV

        # Perform object detection
        results = model(frame)
        annotated_frame = results[0].plot()

        # Display
        cv2.imshow("YOLOv8 Detection with ZED", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
zed.close()
cv2.destroyAllWindows()
