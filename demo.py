import streamlit as st
import cv2
import cvzone
from cvzone.PoseModule import PoseDetector
import numpy as np
from PIL import Image

# Streamlit Title
st.title("Virtual Shirt Try-On")

# Sidebar for Shirt Selection
uploaded_shirt = st.selectbox('Select',['static/shirt/p4.png', 'static/shirt/p5.png', 'static/shirt/p6.png', 'static/shirt/p7.png', 'static/shirt/black.png', 'static/shirt/red.png'])
shirt_images = ["static/shirt/p4.png"]  # Default shirt path if no file uploaded

if uploaded_shirt:
    shirt_path = uploaded_shirt
else:
    shirt_path = shirt_images[0]

# Initialize Camera and Pose Detector
cap = cv2.VideoCapture(0)
detector = PoseDetector()

# Load Shirt Image
img_Shirt = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)
if img_Shirt is None:
    st.error("Error: Unable to load shirt image. Check the file path.")
    st.stop()

# Aspect ratio of the shirt
fixed_ratio = img_Shirt.shape[0] / img_Shirt.shape[1]

# Streamlit Frame
stframe = st.empty()

while cap.isOpened():
    success, img = cap.read()
    if not success:
        st.error("Camera not detected.")
        break

    img = cv2.resize(img, (960, 720), None)
    img = detector.findPose(img, draw=False)
    lmList, bboxInfo = detector.findPosition(img, draw=False, bboxWithHands=False)

    if lmList:
        # Get relevant landmarks: shoulders and hips
        lm11 = lmList[11][:2]  # Left shoulder
        lm12 = lmList[12][:2]  # Right shoulder
        lm23 = lmList[23][:2]  # Left hip
        lm24 = lmList[24][:2]  # Right hip

        # Calculate shirt width based on shoulder distance
        shirt_width = int(abs(lm12[0] - lm11[0]) * 1.8)
        shirt_height = int(shirt_width * fixed_ratio * 1.15)

        # Validate dimensions
        if shirt_width > 0 and shirt_height > 0:
            resized_shirt = cv2.resize(img_Shirt, (shirt_width, shirt_height))

            # Calculate top-left position for the shirt
            shoulder_center_x = (lm11[0] + lm12[0]) // 2
            shoulder_center_y = (lm11[1] + lm12[1]) // 2
            hip_center_y = (lm23[1] + lm24[1]) // 2
            chest_center_y = shoulder_center_y + (hip_center_y - shoulder_center_y) // 3
            
            top_left_x = shoulder_center_x - shirt_width // 2
            top_left_y = chest_center_y - shirt_height // 2 + int(-0.25 * shirt_height)

            # Overlay the shirt image on the frame
            try:
                img = cvzone.overlayPNG(img, resized_shirt, [top_left_x, top_left_y])
            except Exception as e:
                st.error(f"Error overlaying shirt: {e}")
        else:
            st.warning("Invalid dimensions for resizing.")
    else:
        st.warning("Pose not detected. Ensure the subject is visible in the frame.")

    # Convert BGR image to RGB for Streamlit display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    stframe.image(img_rgb, channels="RGB", use_column_width=True)

cap.release()
st.info("Application stopped. Thank you for using!")
