import os
import io
import cv2
import numpy as np
import urllib.request
from rembg import remove
from PIL import Image

# --- 1. AUTO-DOWNLOAD MODELS (First Run Only) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model 1: YuNet (Modern Face Detector)
YUNET_PATH = os.path.join(BASE_DIR, "face_detection_yunet_2023mar.onnx")
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

# Model 2: LBF (Facial Landmarks)
LBF_PATH = os.path.join(BASE_DIR, "lbfmodel.yaml")
LBF_URL = (
    "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"
)


def download_file(url, path):
    if not os.path.exists(path):
        print(f"Downloading required AI model: {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)
        print("Download complete!")


# Ensure models exist before loading
download_file(YUNET_URL, YUNET_PATH)
download_file(LBF_URL, LBF_PATH)

# --- 2. INITIALIZE MODELS ---
# Initialize YuNet Face Detector
face_detector = cv2.FaceDetectorYN.create(
    model=YUNET_PATH,
    config="",
    input_size=(320, 320),
    score_threshold=0.8,
    nms_threshold=0.3,
    top_k=5000,
)

# Initialize LBF Landmark Detector
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(LBF_PATH)


def process_passport_photo(image_bytes):
    """
    Advanced Passport Photo Processor
    """
    # Load image for YuNet detection
    np_arr = np.frombuffer(image_bytes, np.uint8)
    cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Load image for rembg
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    # Remove background
    no_bg_image = remove(input_image)
    bbox = no_bg_image.getbbox()
    if bbox:
        no_bg_image = no_bg_image.crop(bbox)
        bbox_left, bbox_top = bbox[0], bbox[1]
    else:
        bbox_left, bbox_top = 0, 0

    # Set target canvas size (35x45mm at 300 DPI)
    target_width, target_height = 413, 531
    paste_x, paste_y = 0, 0
    final_img = no_bg_image

    # --- DETECT FACE (YuNet) ---
    h, w, _ = cv_img.shape
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(cv_img)

    if faces is not None and len(faces) > 0:
        # Get the largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w_face, h_face = face[:4]

        # Format bounding box for LBF
        faces_array = np.array([[int(x), int(y), int(w_face), int(h_face)]])

        # Fit landmarks
        ok, landmarks = facemark.fit(cv_img, faces_array)

        if ok and len(landmarks) > 0:
            points = landmarks[0][0]  # Shape should be (68, 2)

            # SAFETY CHECK: Ensure we actually got 68 points
            if len(points) >= 68:
                # Landmark indices: 36-41 (Left Eye), 42-47 (Right Eye), 8 (Chin)
                left_eye = np.mean(points[36:42], axis=0)
                right_eye = np.mean(points[42:48], axis=0)
                chin = points[8]

                eye_center_y = (left_eye[1] + right_eye[1]) / 2
                eye_center_x = (left_eye[0] + right_eye[0]) / 2

                # --- BIOMETRIC MATH ---
                target_head_height = int(target_height * 0.75)
                eye_to_chin = chin[1] - eye_center_y

                if eye_to_chin <= 0:
                    eye_to_chin = h_face * 0.5

                estimated_head_height = eye_to_chin * 2.2
                scale = target_head_height / estimated_head_height

                # Resize
                new_w = int(no_bg_image.width * scale)
                new_h = int(no_bg_image.height * scale)
                resized_img = no_bg_image.resize(
                    (new_w, new_h), Image.Resampling.LANCZOS
                )

                # Eye placement at 45% from top
                target_eye_y = int(target_height * 0.45)

                # Calculate coordinates in the resized image
                resized_eye_y = int((eye_center_y - bbox_top) * scale)
                resized_eye_x = int((eye_center_x - bbox_left) * scale)

                # Calculate paste offsets
                paste_y = target_eye_y - resized_eye_y
                paste_x = (target_width // 2) - resized_eye_x

                final_img = resized_img
            else:
                # Landmark model failed to find 68 points, use fallback
                print(
                    f"Warning: Landmark model returned {len(points)} points instead of 68. Using fallback."
                )
                final_img.thumbnail(
                    (target_width, target_height), Image.Resampling.LANCZOS
                )
                paste_x = (target_width - final_img.width) // 2
                paste_y = int(target_height * 0.1)  # Move slightly up
        else:
            # Facemark fit failed completely
            final_img.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
            paste_x = (target_width - final_img.width) // 2
            paste_y = int(target_height * 0.1)
    else:
        # Fallback if no face detected at all
        final_img.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
        paste_x = (target_width - final_img.width) // 2
        paste_y = (target_height - final_img.height) // 2

    # --- COMPOSITE ---
    background = Image.new("RGB", (target_width, target_height), (255, 255, 255))
    if final_img.mode == "RGBA":
        background.paste(final_img, (paste_x, paste_y), final_img)
    else:
        background.paste(final_img, (paste_x, paste_y))

    return background
