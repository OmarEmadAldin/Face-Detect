#!/usr/bin/env python3
import cvzone
import cv2
from ultralytics import YOLO


class FaceDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.3):
        """Initialize the face detection model."""
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_faces(self, frame):
        """Detect faces and return the largest bounding box."""
        face_result = self.model.predict(frame, conf=self.conf_threshold)
        largest_area = 0
        largest_bbox = None

        for info in face_result:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                area = w * h

                # Find the largest face by area
                if area > largest_area:
                    largest_area = area
                    largest_bbox = (x1, y1, x2, y2)

        return largest_bbox


class FrameProcessor:
    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        """Initialize frame processor."""
        self.frame_width = frame_width
        self.frame_height = frame_height

    def resize_frame(self, frame):
        """Resize the frame to the desired dimensions."""
        return cv2.resize(frame, (self.frame_width, self.frame_height))

    def draw_center_lines(self, frame):
        """Draw horizontal and vertical center lines on the frame."""
        height, width, _ = frame.shape
        frame_center_x = width // 2
        frame_center_y = height // 2

        # Draw center lines
        cv2.line(frame, (0, frame_center_y), (width, frame_center_y), (0, 255, 0), 2)
        cv2.line(frame, (frame_center_x, 0), (frame_center_x, height), (0, 255, 0), 2)
        cv2.circle(frame, (frame_center_x, frame_center_y), 10, (255, 0, 0), -1)  # Blue circle

        return frame, (frame_center_x, frame_center_y)


class CameraController:
    def __init__(self, camera_id: int = 0, model_path: str = '/path/to/yolov8n-face.pt'):
        """Initialize camera controller with face detection and frame processing."""
        self.cap = cv2.VideoCapture(camera_id)
        self.face_detector = FaceDetector(model_path)
        self.frame_processor = FrameProcessor()

    def start(self):
        """Start processing the camera feed."""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Resize and draw center lines
            frame = self.frame_processor.resize_frame(frame)
            frame, (frame_center_x, frame_center_y) = self.frame_processor.draw_center_lines(frame)

            # Perform face detection and track the largest face
            largest_bbox = self.face_detector.detect_faces(frame)

            if largest_bbox:
                x1, y1, x2, y2 = largest_bbox
                w, h = x2 - x1, y2 - y1

                # Draw bounding box around the largest face
                cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3)

                # Calculate the center of the detected face
                face_center_x = (x1 + x2) // 2
                face_center_y = (y1 + y2) // 2
                cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 0, 255), -1)  # Red circle

                # Calculate the error between the face and the center of the frame
                error_x = face_center_x - frame_center_x
                error_y = face_center_y - frame_center_y
                self.display_error(error_x, error_y)
            else:
                print("No face detected.")

            # Show the processed frame
            cv2.imshow('Frame', frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def display_error(self, error_x, error_y):
        """Display errors and suggest motor movements."""
        print(f"Error in X: {error_x}")
        if error_x > 0:
            print("Move Left Motor")
        else:
            print("Move Right Motor")

        print(f"Error in Y: {error_y}")

    def cleanup(self):
        """Release resources."""
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_controller = CameraController(camera_id=0, model_path='/home/omar_ben_emad/Github_repos/Face-Detect/Modern/Python_codes/yolov8n-face.pt')
    camera_controller.start()
