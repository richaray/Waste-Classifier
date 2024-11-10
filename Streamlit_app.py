import streamlit as st
import cv2
import tempfile
from wastee import wastee
from Classifier import classify_object, get_model_accuracy

# Streamlit App Title
st.title("Live Object Detection and Classification")

# Start webcam and run detection
st.write("Click the button below to start the webcam for live object detection:")

# Button to start detection
if st.button("Start Object Detection"):
    st.write("Starting the webcam...")

    # Use temporary file to handle video stream
    temp_video_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)

    # Call the object detection function
    detected_object = wastee()

    if detected_object:
        st.write(f"Detected Object: {detected_object}")

        # Classify the detected object
        result, object_name = classify_object(detected_object)

        if result == 1:
            st.success(f"The object '{object_name}' is recyclable. BLUE BIN ")
        else:
            st.warning(f"The object '{object_name}' is non-recyclable. RED BIN")
    else:
        st.write("No valid object detected.")

# Display the accuracy of the trained model
st.write("Evaluating the model accuracy:")
accuracy = get_model_accuracy()
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
