import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
import gradio as gr
import tempfile
import os

base_dir = os.getcwd()
saved_model_dir = os.path.join(base_dir, 'saved_model')

# Loading the trained CNN model
model = load_model(saved_model_dir)


# Initializing the MTCNN face detector
detector = MTCNN()


# Making a function for fetching roi coordinates, performing classification and displaying image having detection
def classify_faces(img):
    faces = detector.detect_faces(img)
    sleepy_faces = 0

    for face in faces:
        x, y, w, h = face['box']
        x1 = face['keypoints']['left_eye'][0]
        y1 = face['keypoints']['left_eye'][1]
        x2 = face['keypoints']['right_eye'][0]
        y2 = face['keypoints']['right_eye'][1]

        # Calculating the distance between the eyes
        eye_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if abs(x2 - x1) > abs(y2 - y1):
            # For larger horizontal distances between eyes
            roi_w = int(5 / 3 * eye_distance)
            roi_h = int(2 / 3 * eye_distance)
        else:
            # For larger vertical distances between eyes
            roi_w = int(2 / 3 * eye_distance)
            roi_h = int(5 / 3 * eye_distance)

        # Calculating the center point between the eyes
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Adjusting ROI coordinates to keep the center point between the eyes (It essentially grabs the top left
        # coordinate of the roi box)
        roi_x = int(center_x - roi_w / 2)
        roi_y = int(center_y - roi_h / 2)

        # Ensuring the ROI is within image boundaries
        roi_x = max(0, roi_x)
        roi_y = max(0, roi_y)
        roi_w = min(roi_w, img.shape[1] - roi_x)
        roi_h = min(roi_h, img.shape[0] - roi_y)

        crop = img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        # Preprocessing the cropped face image as required by your model
        crop_resized = cv2.resize(crop, (224, 224))  # Assuming your model expects 224x224 input
        crop_resized = crop_resized.astype('float32') / 255.0  # Normalize if required
        crop_resized = np.expand_dims(crop_resized, axis=0)  # Add batch dimension

        prediction = model.predict(crop_resized)
        label = 'Awake' if prediction[0][0] < 0.5 else 'Sleepy'

        if label == 'Sleepy':
            sleepy_faces += 1
            # Drawing bounding box around drowsy face
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Putting text label above the bounding box
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Displaying the count of sleepy faces detected
    cv2.putText(img, f'Sleepy faces: {sleepy_faces}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img, sleepy_faces


def process_image(image_path):
    
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Unable to load image from {image_path}")

    # Converting BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resizing the image to fit within a fixed window size while maintaining aspect ratio
    max_display_size = 800  # Maximum width or height for displaying the image
    height, width, _ = img_rgb.shape
    if max(height, width) > max_display_size:
        if height > width:
            new_height = max_display_size
            new_width = int(width * (max_display_size / height))
        else:
            new_width = max_display_size
            new_height = int(height * (max_display_size / width))
        img_rgb = cv2.resize(img_rgb, (new_width, new_height))

    # Classifying faces and retrieving image with bounding boxes
    img_with_boxes, sleepy_faces = classify_faces(img_rgb)

    # Converting back to BGR for saving with OpenCV
    img_with_boxes_bgr = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)

    return img_with_boxes_bgr, f'Sleepy faces detected: {sleepy_faces}'


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    max_sleepy_faces = 0

    # Obtaining frame dimensions and FPS from the video capture
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Converting the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_with_boxes, sleepy_faces = classify_faces(frame_rgb)
        frames.append(frame_with_boxes)

        # Updating maximum sleepy faces count if current frame has more
        if sleepy_faces > max_sleepy_faces:
            max_sleepy_faces = sleepy_faces

    cap.release()

    # Saving the processed video to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(temp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    for frame in frames:
        # Converting the frame back to BGR for saving
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()

    return temp_file.name, f'The maximum number of sleepy faces detected in the entire video is: {max_sleepy_faces}'


def image_interface(image):
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    image.save(temp_input.name)
    result_image, detection_info = process_image(temp_input.name)
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    cv2.imwrite(temp_output.name, result_image)
    return temp_output.name, detection_info


def video_interface(video_path):
    result_video, detection_info = process_video(video_path)
    return result_video, detection_info

image_container = gr.Interface(fn=image_interface, inputs=gr.Image(type="pil"), outputs=[gr.Image(), gr.Text()])
video_container = gr.Interface(fn=video_interface, inputs=gr.Video(), outputs=[gr.Video(), gr.Text()])

with gr.Blocks() as container:
    gr.Markdown("# Sleep Detection")
    gr.Markdown("### Made by Joy Biswas")
    gr.TabbedInterface([image_container, video_container], ["Image Detection", "Video Detection"])

container.launch()
