from flask import Flask, Response, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import cv2
import numpy as np
import math
from gaze_tracking import GazeTracking
import time

app = Flask(__name__)

# ✅ Fix CORS for WebSockets & API requests
CORS(app, resources={r"/*": {"origins": "*"}})  # Allows all origins
socketio = SocketIO(app, cors_allowed_origins="*")  # ✅ Fix WebSocket CORS

# Initialize Gaze Tracking
gaze = GazeTracking()
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load Facial Landmark Model
LBFmodel = "lbfmodel.yaml"
landmark_detector = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)

# Start Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set Width
cap.set(4, 720)  # Set Height

# Global Variables for Attention Status
text = "Neutral 🙂"
gaze_direction = "Center"
blink_detected = False

# 3D Model Points
model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
])


def detect_faces(img):
    """Detect faces using OpenCV"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=2, minSize=(20, 20))
    return faces, gray


def pose_estimate(img, landmarks):
    """Estimate head pose using facial landmarks"""
    size = img.shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    shape = np.array(landmarks, dtype=np.float32).astype(np.uint)
    image_points = np.array([
        shape[0][0][30],  # Nose tip
        shape[0][0][8],  # Chin
        shape[0][0][36],  # Left eye left corner
        shape[0][0][45],  # Right eye right corner
        shape[0][0][48],  # Left Mouth corner
        shape[0][0][54]  # Right mouth corner
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP
    )

    (nose_end_point2D, _) = cv2.projectPoints(
        np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs
    )

    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    try:
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        angle = int(math.degrees(math.atan(m)))
    except ZeroDivisionError:
        angle = 90

    cv2.line(img, p1, p2, (0, 255, 255), 2)

    return -45 < angle < 45  # True if the person is looking straight


def track_attention():
    """Continuously track attention and send real-time updates via WebSocket."""
    global text, gaze_direction, blink_detected

    while True:
        success, img = cap.read()
        if not success:
            break

        faces, gray = detect_faces(img)
        gaze.refresh(img)
        gaze_direction = "Center" if gaze.is_center() else "Left" if gaze.is_left() else "Right"
        blink_detected = gaze.is_blinking()
        attention = gaze.is_center()

        boolout = False
        try:
            _, landmarks = landmark_detector.fit(gray, faces)
            boolout = pose_estimate(img, landmarks) and attention
        except Exception as e:
            print(e)

        # Determine attention level
        if boolout:
            text = "Definitely Focused 🎯"
        elif attention:
            text = "Somewhat Focused 🤔"
        else:
            text = "Distracted 🤣"

        # Send real-time data to frontend via WebSockets
        socketio.emit("attention_data", {
            "focus_status": text,
            "gaze_direction": gaze_direction,
            "blinking": "Yes" if blink_detected else "No"
        })

        time.sleep(0.5)  # Update every 500ms (low latency)


@app.route('/video_feed')
def video_feed():
    """Stream live video with attention tracking overlay."""

    def generate_frames():
        while True:
            success, img = cap.read()
            if not success:
                break

            gaze.refresh(img)
            img = gaze.annotated_frame()

            _, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    response = Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers.add("Access-Control-Allow-Origin", "*")  # ✅ FIX
    response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
    return response


@app.route('/api/status', methods=['GET'])
def get_status():
    """Return real-time attention span data"""
    global text, gaze_direction, blink_detected
    response = jsonify({
        "focus_status": text,
        "gaze_direction": gaze_direction,
        "blinking": blink_detected
    })
    response.headers.add("Access-Control-Allow-Origin", "*")  # 🔹 Ensures CORS works
    response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
    return response


if __name__ == '__main__':
    socketio.start_background_task(target=track_attention)  # Start real-time tracking
    socketio.run(app, debug=True, host="0.0.0.0", port=4200)
