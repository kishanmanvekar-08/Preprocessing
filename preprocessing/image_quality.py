
import cv2
import numpy as np

def check_image_quality(image):
    if image is None or image.size == 0:
        return False, {"reason": "Invalid image"}

    height, width = image.shape[:2]

    if width < 160 or height < 160:
        return False, {"reason": "Image resolution is too low"}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur measurement
    blur_score = cv2.Laplacian(
        gray, cv2.CV_64F
    ).var()

    # Brightness measurement
    brightness = np.mean(gray)

    issues = []

    if blur_score < 60:
        issues.append("Image is blurry")

    if brightness < 35:
        issues.append("Image is too dark")

    if brightness > 225:
        issues.append("Image is too bright")

    accepted = len(issues) == 0

    report = {
        "width": width,
        "height": height,
        "blur_score": round(float(blur_score), 2),
        "brightness": round(float(brightness), 2),
        "issues": issues
    }

    return accepted, report
