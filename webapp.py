"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
from PIL import Image
import datetime

import torch
import cv2
import numpy as np
import tensorflow as tf
from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, Response
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import time

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

# ... other imports and your existing code ...


app = Flask(__name__)





@app.route("/")
def hello_world():
    return render_template('index.html')


# function for accessing rtsp stream
# @app.route("/rtsp_feed")
# def rtsp_feed():
    # cap = cv2.VideoCapture('rtsp://admin:hello123@192.168.29.126:554/cam/realmonitor?channel=1&subtype=0')
    # return render_template('index.html')


# Function to start webcam and detect objects

@app.route("/webcam_feed")
def webcam_feed():
    #source = 0
    # cap = cv2.VideoCapture(0)
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

# function to get the frames from video (output video)

def get_frame(filename_frame):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    filename = filename_frame #predict_img.imgpath
    image_path = folder_path+'/'+latest_subfolder+'/'+filename    
    video = cv2.VideoCapture(image_path)  # detected video path
    #video = cv2.VideoCapture("video.mp4")
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)   
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')   
        time.sleep(0.1)  #control the frame rate to display one frame every 100 milliseconds: 


# function to display the detected objects video on html page
@app.route("/video_feed")
def video_feed(filename):
    return Response(get_frame(filename),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



#The display function is used to serve the image or video from the folder_path directory.
@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    directory = folder_path+'/'+latest_subfolder
    print("printing directory: ",directory)  
    filename = filename #predict_img.imgpath
    file_extension = filename.rsplit('.', 1)[1].lower()
    #print("printing file extension from display function : ",file_extension)
    environ = request.environ
    if file_extension == 'jpg':      
        return send_from_directory(directory,filename,environ)

    elif file_extension == 'mp4':
        return render_template('index.html')

    else:
        return "Invalid file format"

    
@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath,'uploads',f.filename)
            print("upload folder is ", filepath)
            f.save(filepath)
            
            predict_img.imgpath = f.filename
            print("printing predict_img :::::: ", predict_img)

            process = Popen(["python", "detect.py", '--source', filepath, "--weights","best.pt"], shell=True)
            process.wait()

            file_extension = f.filename.rsplit('.', 1)[1].lower()    
            if file_extension == 'jpg':
                return display(f.filename)
            elif file_extension == 'mp4':
                return video_feed(f.filename)

            
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    image_path = folder_path+'/'+latest_subfolder+'/'+f.filename 
    return render_template('index.html', image_path=image_path)
    #return "done"


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]  # Get the original image dimensions
    target_h, target_w = new_shape  # Target dimensions

    # Calculate the new size while maintaining the aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize the image using the calculated dimensions
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create a blank canvas of the target size and fill it with the specified color
    canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)

    # Calculate the position to paste the resized image on the canvas
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    # Paste the resized image onto the canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = img_resized

    return canvas





def generate_frames():
    camera = cv2.VideoCapture(0)
    
    # Load YOLOv7 model and configuration
    weights = "best.pt"  # Replace with your YOLOv7 weights file
    img_size = [640, 640] # Set your desired image size

    # Initialize
    # device = select_device('0')  # Change '0' to the GPU device you want to use
    # half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location='cpu')  # load FP32 model
    stride = int(model.stride.max())  # model stride

    # Set Dataloader
    # dataset = LoadImages(source='0', img_size=img_size, stride=stride)  # Use '0' for webcam

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    conf_thres = 0.25
    iou_thres = 0.45

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            im0s = frame.copy()
            img = letterbox(im0s, new_shape=img_size) #[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to('cpu')
            img = img.float() / 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():
                pred = model(img, augment=False)[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres)[0]
            n_detected = len(pred)
            t2 = time_synchronized()

            for *xyxy, conf, cls in pred:
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0s, label=label, color=colors[int(cls)], line_thickness=3)

                # add number of detected objects
                detected_xyxy = [torch.tensor(20., device='cpu'), torch.tensor(50., device='cpu'), torch.tensor(20., device='cpu'), torch.tensor(50., device='cpu')]
                plot_one_box(detected_xyxy, im0s, label=f'Jumlah Kehadiran: {n_detected}', color=[0, 0, 255], line_thickness=3)

            ret, buffer = cv2.imencode('.jpg', im0s)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ... the rest of your code ...



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov7 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    model = torch.hub.load('.', 'custom','best.pt', source='local')
    model.eval()
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat

