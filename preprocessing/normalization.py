
import cv2
import numpy as np

def normalize_image(image, size=(224, 224)):

    target_w, target_h = size

    h, w = image.shape[:2]

    # Maintain original aspect ratio
    scale = min(
        target_w / w,
        target_h / h
    )

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # Create white square background
    canvas = np.full(
        (target_h, target_w, 3),
        255,
        dtype=np.uint8
    )

    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2

    canvas[
        y:y+new_h,
        x:x+new_w
    ] = resized

    # Convert OpenCV BGR to RGB
    final_image = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2RGB
    )

    return final_image
