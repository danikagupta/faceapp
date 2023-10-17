import streamlit as st
import face_recognition
import numpy as np

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.


def main():
    st.title("Face Recognition App")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Classifying...")
        
        # Load the uploaded image file
        uploaded_image = face_recognition.load_image_file(uploaded_file)
        
        # Get face locations
        face_locations = face_recognition.face_locations(uploaded_image)
        
        # Display face locations
        for face_location in face_locations:
            top, right, bottom, left = face_location
            st.write(f"A face is located at pixel location Top: {top}, Left: {left}, Bottom: {bottom}, Right: {right}")

if __name__ == "__main__":
    main()


st.markdown("# Page where users will be able to set up their profiles")