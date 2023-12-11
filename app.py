import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
import streamlit as st
import cv2
import os
import numpy as np

@st.cache_resource
def get_file_path(uploaded_file):
    # Save the uploaded file to the temporary directory
    file_path = "./temp/clip"
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    
    return file_path

@st.cache_resource
def model_loader(model_path):
    model = keras.models.load_model(model_path)
    return model

def pred_func1(video_path, model):
    ans = 0
    logs = []
    result = ""
    capture_video = cv2.VideoCapture(video_path)
    writer = None
    (width,height) = (None,None)

    while True:
        (taken,frame) = capture_video.read()
        if not taken:
            break
        if width is None or height is None:
            (width,height) = frame.shape[:2]

        frame = cv2.resize(frame,(256,256)).astype("float32")
        preds = model.predict(np.expand_dims(frame,axis=0), verbose=0)[0]
        logs.append(preds)
        ans = max(ans,preds)
        # results = np.array(queue).mean(axis=0)
        # i = np.argmax(results)
        
    if ans > 1e-25:
        result = "Foul"
    else:
        result = "Clean"
    logs = np.array(logs)
    logs = logs.ravel()
    return result,ans,logs

def pred_func2(video_path, model):
    ans = 0
    logs = []
    result = ""
    capture_video = cv2.VideoCapture(video_path)
    writer = None
    (width,height) = (None,None)

    while True:
        (taken,frame) = capture_video.read()
        if not taken:
            break
        if width is None or height is None:
            (width,height) = frame.shape[:2]

        frame = cv2.resize(frame,(256,256)).astype("float32")
        gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        gray = np.expand_dims(gray,axis=2)
        preds = model.predict(np.expand_dims(gray,axis=0), verbose=0)[0]
        logs.append(preds)
        ans = max(ans,preds)

    if(ans<0.29):
        result = "Clean"
    else:
        result = "Foul"
    logs = np.array(logs)
    logs = logs.ravel()
    return result,ans,logs

def main():
    st.title("Virtual Assistant Referee")
    st.subheader("Player Contact and Foul Detection, using AI based video analysis.")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov","3gp"])
    
    st.sidebar.title("Models")
    model_version = st.sidebar.radio("Select Model", ("V1 (RGB Rendering)", "V2 (Grayscale Rendering)"))


    if uploaded_file is not None:
        file_path = get_file_path(uploaded_file)

        if model_version == "V1 (RGB Rendering)":
            st.subheader("V1: RGB Rendering")
            st.video(uploaded_file)
            model = model_loader('./model_1.h5')
            result,ans,logs = pred_func1(file_path,model)
            st.subheader(f"Prediction: {result}")
            # st.subheader(ans)
        else:
            st.subheader("V2: Grayscale Rendering")
            st.video(uploaded_file)
            model = model_loader('./model_2.h5')
            result,ans,logs = pred_func2(file_path,model)
            st.subheader(f"Prediction: {result}")
            # st.subheader(ans)

if __name__ == "__main__":
    main()
