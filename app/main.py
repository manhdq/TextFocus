import requests
import json
import numpy as np
from io import BytesIO
from PIL import Image

import torch
import streamlit as st

import utils


URL_DICT = {
    "CTW1500": "http://127.0.0.1:6000", 
    "CTW-China": "http://127.0.0.1:6001", 
    "Total-text": "http://127.0.0.1:6002", 
    "ICDAR2015": "http://127.0.0.1:6003",
}

HEADERS = {
    "Content-Type": "application/json",
}


st.set_page_config(layout="wide")
st.title("TextFocus: Efficient Text Detection App 📷")


## Function
def show_uploaded_image():
    global img_is_uploaded
    img = Image.open(img_upload)
    col1.image(img)
    img_is_uploaded = True

def convert_image(img):
    buf = BytesIO()
    try:
        img.save(buf, format="JPEG")
    except:
        img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def convert_json(data):
    json_data = json.dumps(data, indent=4)
    json_bytes = json_data.encode("utf-8")
    return json_bytes

def text_detection_process():
    img_pil = Image.open(img_upload).convert("RGB")
    img = np.array(img_pil)
    
    img_base64 = utils.image_to_base64(img_pil)

    json_data = {
        "img_base64": img_base64
    }

    response = requests.post(url=f"{URL}/text_focus_detect",
                            json=json_data)

    if response.status_code == 202:
        torch.cuda.empty_cache()
        json_results = response.json()
        bboxes = json_results["bboxes"]
        points_group = []
        for bbox in bboxes:
            points_group.append(utils.bboxes_list_to_kpts_array(bbox))
        img = utils.draw_detect(img, points_group, (0, 255, 0))
    else:
        print('Request failed with status code:', response.status_code)
        print('Response:', response.text)
        return

    col2.image(img)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download results image", convert_image(Image.fromarray(img)), "output.jpg", "image/jpeg")
    st.sidebar.download_button("Download json data", convert_json(json_results), file_name="data.json", mime="application/json")


## Setup sidebar
img_is_uploaded = False
json_results = None
url_option = st.sidebar.selectbox(
    "Select weight for efficient text detection",
    ("CTW1500", "CTW-China", "Total-text", "ICDAR2015"),
)
URL = URL_DICT[url_option]
st.sidebar.markdown("\n")
st.sidebar.write("## Upload and download :gear:")
img_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
start_process = st.sidebar.button("Start process")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

## Setup visualize
col1, col2 = st.columns(2)

col1.write("Original Image :camera:")
col2.write("Image with detection results :wrench:")

if img_upload is not None:
    if img_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
        img_is_uploaded = False
    else:
        show_uploaded_image()
else:
    img_is_uploaded = False

if start_process:
    if img_is_uploaded:
        text_detection_process()
    else:
        st.error("Please upload an image before process the model!")