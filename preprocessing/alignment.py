
import cv2
import numpy as np

def align_object(image, mask, target="vertical"):

    ys, xs = np.where(mask > 0)

    if len(xs) < 20:
        return image, mask, 0.0

    points = np.column_stack((xs, ys)).astype(np.float32)

    # Find the object's main direction using PCA
    mean, eigenvectors = cv2.PCACompute(
        points, mean=None
    )

    major_axis = eigenvectors[0]

    angle = np.degrees(
        np.arctan2(major_axis[1], major_axis[0])
    )

    if target == "vertical":
        rotation = 90 - angle
    elif target == "horizontal":
        rotation = -angle
    else:
        raise ValueError("Invalid target")

    rotation = ((rotation + 180) % 360) - 180

    if rotation > 90:
        rotation -= 180
    elif rotation < -90:
        rotation += 180

    h, w = image.shape[:2]

    center = (w / 2, h / 2)

    matrix = cv2.getRotationMatrix2D(
        center, rotation, 1.0
    )

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    matrix[0, 2] += new_w / 2 - center[0]
    matrix[1, 2] += new_h / 2 - center[1]

    aligned = cv2.warpAffine(
        image, matrix, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    aligned_mask = cv2.warpAffine(
        mask, matrix, (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return aligned, aligned_mask, rotation
