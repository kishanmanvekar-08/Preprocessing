
import cv2
import numpy as np

def extract_roi(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(
        gray, (5, 5), 0
    )

    # Try to separate foreground from background
    _, mask1 = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    _, mask2 = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (7, 7)
    )

    candidates = []

    for mask in [mask1, mask2]:

        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE,
            kernel, iterations=2
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)

            if area > image.shape[0] * image.shape[1] * 0.01:
                candidates.append((area, contour, mask))

    if not candidates:
        raise ValueError("Object not detected. Try a plain background.")

    _, contour, mask = max(
        candidates, key=lambda item: item[0]
    )

    x, y, w, h = cv2.boundingRect(contour)

    # Add a small margin around the object
    px = int(w * 0.06)
    py = int(h * 0.06)

    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(image.shape[1], x + w + px)
    y2 = min(image.shape[0], y + h + py)

    roi = image[y1:y2, x1:x2]
    roi_mask = mask[y1:y2, x1:x2]

    return roi, roi_mask, (x1, y1, x2-x1, y2-y1)
