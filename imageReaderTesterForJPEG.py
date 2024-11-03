import cv2
from PIL import Image
import pytesseract
import numpy as np

def preprocess_image(image):
    # Resize the image to a larger size
    resized_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Denoise using Non-Local Means Denoising
    denoised_image = cv2.fastNlMeansDenoising(resized_image, None, h=6, searchWindowSize=21, templateWindowSize=7)

    # Bilateral Filtering for noise reduction
    filtered_image = cv2.bilateralFilter(denoised_image, 9, 75, 75)

    # Convert to grayscale
    gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_image = clahe.apply(gray)

    # Apply adaptive thresholding for binarization
    _, binary_image = cv2.threshold(contrast_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological operations (erosion and dilation)
    kernel = np.ones((3, 3), np.uint8)
    binary_image = cv2.erode(binary_image, kernel, iterations=1)
    binary_image = cv2.dilate(binary_image, kernel, iterations=1)

    return contrast_image, binary_image

def extract_text(contrast_image, binary_image):
    # Find contours of text regions
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask to overlay on the original image
    mask = np.zeros_like(binary_image)

    # Iterate through contours and draw on mask
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    # Extract text regions using the mask
    text_regions = cv2.bitwise_and(contrast_image, contrast_image, mask=mask)

    # Apply OCR on text regions
    text = pytesseract.image_to_string(Image.fromarray(text_regions))

    return text

def correct_skew(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours
    contours = [contour for contour in contours if cv2.contourArea(contour) > 100]

    # Find the main contour (assumed to be the text region)
    main_contour = max(contours, key=cv2.contourArea)

    # Get the orientation angle of the bounding box
    _, _, angle = cv2.fitEllipse(main_contour)

    # Correct skew using rotation
    rotated_image = rotate_image(image, angle)

    return rotated_image

def rotate_image(image, angle):
    # Get image center and rotation matrix
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return rotated_image



# Example usage
image_path = r"C:\Users\harve\Downloads\RaspberryPi_and_Code\ackaton2024\SchneiderPrizeScreenshots\03902.PNG"
original_image = cv2.imread(image_path)

if original_image is None:
    print(f"Error: Could not load image at {image_path}")
    exit()

# Correct skew
corrected_image = correct_skew(original_image)

# Preprocess corrected image
contrast_image, binary_image = preprocess_image(corrected_image)

# Extract text from preprocessed image
extracted_text = extract_text(contrast_image, binary_image)
print(extracted_text)
