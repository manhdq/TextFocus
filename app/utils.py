import base64
import cv2
import numpy as np
from PIL import Image
from io import BytesIO


def image_to_base64(img_pil):
    # Convert the image to a byte stream
    image_byte_array = BytesIO()
    img_pil.save(image_byte_array, format="JPEG")
    
    # Encode the byte stream as base64
    img_base64 = base64.b64encode(image_byte_array.getvalue()).decode('utf-8')

    return img_base64


def base64_to_image(base64_string):
    # Decode the Base64 string to bytes
    image_bytes = base64.b64decode(base64_string)
    
    # Create an Image object from the bytes
    image = Image.open(BytesIO(image_bytes))

    return image


def bboxes_list_to_kpts_array(bboxes):
    assert len(bboxes) % 2 == 0, "number of item in list has to be even!"
    num_kpts = len(bboxes) // 2

    kpts = np.array(bboxes).reshape(num_kpts, 2)
    return kpts


def draw_detect(image, points_group, boundary_color):
    '''
    Visualize bbox, landmarks and focus mask
    '''
    new_image = image.copy()
    image_height, image_width = new_image.shape[:2]

    # Draw boundary
    boundary_size = max(min(image_height, image_width) // 1000, 1) + 1
    for points in points_group:
        new_image = cv2.polylines(new_image,
                        [points.astype(np.int32)], True, boundary_color, boundary_size)  # Draw last level
    
    return new_image