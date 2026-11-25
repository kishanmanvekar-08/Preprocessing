
import cv2
import matplotlib.pyplot as plt

from preprocessing.image_quality import check_image_quality
from preprocessing.roi import extract_roi
from preprocessing.alignment import align_object
from preprocessing.normalization import normalize_image


import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from preprocessing.image_quality import check_image_quality
from preprocessing.roi import extract_roi
from preprocessing.alignment import align_object
from preprocessing.normalization import normalize_image

# Get the location of this Python file
BASE_DIR = Path(__file__).resolve().parent

# Define image and output paths
image_path = BASE_DIR / "test_images" / "sample.jpg"
output_dir = BASE_DIR / "output"

# Create output folder if it does not exist
output_dir.mkdir(parents=True, exist_ok=True)

# Load image
image = cv2.imread(str(image_path))

if image is None:
    raise ValueError(f"Image not found or cannot be read: {image_path}")

print("Image loaded successfully!")

# STEP 1: Image Quality Check
accepted, report = check_image_quality(image)

print("Quality report:", report)

if not accepted:
    raise ValueError("Image quality is not acceptable")

# STEP 2: ROI Extraction
roi, mask, bbox = extract_roi(image)

print("ROI bounding box:", bbox)

# STEP 3: Object Alignment
aligned, aligned_mask, angle = align_object(
    roi, mask, target="vertical"
)

print("Rotation angle:", angle)

# STEP 4: Image Normalization
final_image = normalize_image(
    aligned, size=(224, 224)
)

# STEP 5: Save Processed Images
cv2.imwrite(str(output_dir / "roi.jpg"), roi)

cv2.imwrite(str(output_dir / "aligned.jpg"), aligned)

cv2.imwrite(
    str(output_dir / "normalized.jpg"),
    cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
)

print("All processed images saved successfully!")

# STEP 6: Display Results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
plt.title("ROI + Alignment")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(final_image)
plt.title("Normalized Image")
plt.axis("off")

plt.tight_layout()
plt.show()


# STEP 1: Image quality
accepted, report = check_image_quality(image)

print("Quality report:", report)

if not accepted:
    raise ValueError("Image quality is not acceptable")

# STEP 2: ROI extraction
roi, mask, bbox = extract_roi(image)

print("ROI bounding box:", bbox)

# STEP 3: Alignment
aligned, aligned_mask, angle = align_object(
    roi, mask, target="vertical"
)

print("Rotation:", angle)

# STEP 4: Normalization
final_image = normalize_image(
    aligned, size=(224, 224)
)

# Save outputs
cv2.imwrite("output/roi.jpg", roi)
cv2.imwrite("output/aligned.jpg", aligned)

cv2.imwrite(
    "output/normalized.jpg",
    cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
)

# Display results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
plt.title("ROI + Alignment")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(final_image)
plt.title("Normalized Image")
plt.axis("off")

plt.show()
