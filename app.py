from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import cv2
from PIL import Image
import pytesseract
import numpy as np
import os
import re
from firebase_admin import firestore
import firebase_admin
from firebase_admin import credentials, storage
import time
from google.api_core.exceptions import GoogleAPICallError, RetryError
import secrets
import gc
from functools import partial
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import logging
import io
import datetime


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Firebase
cred = credentials.Certificate(r"hackathon2024-d0a30-firebase-adminsdk-1vp6d-e692d343f1.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'hackathon2024-d0a30.appspot.com'
})
db = firestore.client()
bucket = storage.bucket()
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Generate a random secret key
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_PROCESSING_TIME = 30  # Maximum processing time in seconds
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def preprocess_image(image, max_dimension=1500):
    """
    Preprocess image with memory-efficient operations and size limits
    """
    try:
        # Calculate scaling factor if image is too large
        height, width = image.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            width = int(width * scale)
            height = int(height * scale)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            del scale  # Free up memory

        # Convert to grayscale early to reduce memory usage
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        del image  # Free up original image memory

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_image = clahe.apply(gray)
        del gray, clahe  # Free up memory

        # Binarization
        _, binary_image = cv2.threshold(contrast_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return contrast_image, binary_image
    except Exception as e:
        logger.error(f"Error in preprocess_image: {str(e)}", exc_info=True)
        raise


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def safe_process_image(filepath, max_dimension=1500):
    """
    Process image with memory limits and better error handling
    Ensures cropped images are properly saved to uploads folder
    """
    logger.info(f"Starting image processing for {filepath}")
    cropped_filename = None
    
    try:
        # Read image in chunks if possible
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError("Failed to load image")

        logger.info(f"Image loaded. Dimensions: {image.shape[1]}x{image.shape[0]}")
        height, width = image.shape[:2]
        crop_dimensions = {
            'top': 0,
            'bottom': int(height * (5/12)),
            'left': int(width * (1/5)),
            'right': int(width * (4/5))
        }
        cropped_image = image[
            crop_dimensions['top']:crop_dimensions['bottom'],
            crop_dimensions['left']:crop_dimensions['right']
        ]


        # Process image
        contrast_image, binary_image = preprocess_image(image, max_dimension)
        del image  # Free up original image memory
        
        # Calculate crop dimensions (using contrast_image dimensions)

        
        # Create cropped image

        
        # Generate unique filename for cropped image
        original_filename = os.path.basename(filepath)
        cropped_filename = f"cropped_{original_filename}"
        cropped_path = os.path.join(UPLOAD_FOLDER, cropped_filename)
        
        # Ensure the UPLOAD_FOLDER exists
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        # Save cropped image with explicit checks
        success = cv2.imwrite(cropped_path, cropped_image)
        if not success:
            raise IOError(f"Failed to save cropped image to {cropped_path}")
            
        logger.info(f"Cropped image saved successfully to {cropped_path}")
        del cropped_image  # Free up memory

        # Extract and parse text with memory limits
        logger.info("Starting text extraction")
        extracted_text = pytesseract.image_to_string(
            Image.fromarray(contrast_image),
            config='--oem 3 --psm 6'
        )
        del contrast_image, binary_image  # Free up memory
        
        # Parse extracted data with memory limits
        color_data = parse_extracted_data_safely(extracted_text)
        del extracted_text  # Free up memory
        
        return color_data, cropped_filename
        
    except Exception as e:
        logger.error(f"Error in safe_process_image: {str(e)}", exc_info=True)
        # If there was an error and a cropped file was created, try to clean it up
        if cropped_filename:
            try:
                cleanup_path = os.path.join(UPLOAD_FOLDER, cropped_filename)
                if os.path.exists(cleanup_path):
                    os.remove(cleanup_path)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up cropped file: {str(cleanup_error)}")
        raise
    finally:
        # Force garbage collection
        gc.collect()


def save_to_database(sample_id, color_data):
    try:
        # Reference the document in the Firestore collection
        doc_ref = db.collection('color_data').document(sample_id)
        
        # Add the color data to the document
        doc_ref.set(color_data)
        print(f"Data for sample ID {sample_id} successfully saved to the database.")
        
    except Exception as e:
        print(f"Error saving data to the database: {e}")

def safe_db_call(callable, *args, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return callable(*args)
        except (GoogleAPICallError, RetryError) as e:
            logger.error(f"Database attempt {attempt + 1} failed: {str(e)}")
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                raise
    raise RuntimeError("All attempts to call Firestore failed")

@app.route('/search', methods=['POST'])
def search_sample():
    sample_id = request.form['sample_id']
    
    # Reference to the document in Firebase
    doc_ref = db.collection('color_data').document(sample_id)
    
    # Use safe_db_call to handle retries for Firebase requests
    doc = safe_db_call(doc_ref.get)

    if doc.exists:
        data = doc.to_dict()

        # Format color data and delta E values as initially acquired
        color_data = {
            'standard_sample': {
                'RGB': data.get('standard_sample', {}).get('RGB', []),
                'CMYK': data.get('standard_sample', {}).get('CMYK', []),
                'HEX': data.get('standard_sample', {}).get('HEX', ''),
                'CIELAB': data.get('standard_sample', {}).get('CIELAB', []),
                'LCH': data.get('standard_sample', {}).get('LCH', [])
            },
            'test_sample': {
                'RGB': data.get('test_sample', {}).get('RGB', []),
                'CMYK': data.get('test_sample', {}).get('CMYK', []),
                'HEX': data.get('test_sample', {}).get('HEX', ''),
                'CIELAB': data.get('test_sample', {}).get('CIELAB', []),
                'LCH': data.get('test_sample', {}).get('LCH', [])
            }
        }

        # Retrieve delta E values as initially stored
        delta_e_values = {
            'Delta E2000': data.get('delta_e_values', {}).get('Delta E2000', ''),
            'Delta E76': data.get('delta_e_values', {}).get('Delta E76', '')
        }

        # Set image filename based on sample ID
        image_filename = f"cropped_{sample_id}.png"

        # Render the display page with retrieved color data, delta E values, and the image
        return render_template('display.html', sample_id=sample_id, color_data=color_data, delta_e_values=delta_e_values, image_filename=image_filename)
    
    else:
        # Display an error on the index page if the sample ID is not found
        return render_template('index.html', error_message="Sample ID not found. Please try again.")


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

def extract_sample_id(filename):
    # Remove the '.png' extension and return the rest of the filename
    return filename[:-4] if filename.endswith('.PNG') else "Unknown"


def display_data_in_terminal(data):
    """Helper function to debug the parsed data"""
    print("\nParsed Color Data:")
    print("-" * 50)
    
    print("\nStandard Sample:")
    for metric, value in data["standard_sample"].items():
        print(f"{metric}: {value}")
    
    print("\nTest Sample:")
    for metric, value in data["test_sample"].items():
        print(f"{metric}: {value}")
    
    print("\nDelta E Values:")
    for metric, value in data["delta_e_values"].items():
        print(f"{metric}: {value}")
    
    print("-" * 50)


@app.route('/')
def index():
    return render_template('index.html')
def parse_extracted_data_safely(text):
    """
    Memory-efficient parsing of extracted text data with corrected parsing logic
    """
    try:
        data = {
            "standard_sample": {
                "RGB": None,
                "CMYK": None,
                "HEX": None,
                "CIELAB": None,
                "LCH": None,
            },
            "test_sample": {
                "RGB": None,
                "CMYK": None,
                "HEX": None,
                "CIELAB": None,
                "LCH": None,
            },
            "delta_e_values": {
                "Delta E2000": None,
                "Delta E76": None
            }
        }
        
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            # Handle Delta E values
            if "Delta E2000" in line:
                value = line.split("Delta E2000:")[-1].strip()
                data["delta_e_values"]["Delta E2000"] = value
                continue
            elif "Delta E76" in line:
                value = line.split("Delta E76:")[-1].strip()
                data["delta_e_values"]["Delta E76"] = value
                continue

            # Handle color values
            for color_type, label in [
                ("RGB", "RGB:"),
                ("CMYK", "CMYK:"),
                ("HEX", "HEX:"),
                ("CIELAB", "CIELAB:"),
                ("LCH", "LCH(ab):")
            ]:
                if label in line:
                    # Get all occurrences of the label in the line
                    values = line.split(label)[1:]  # Split and remove empty first element
                    values = [v.strip() for v in values]
                    
                    if len(values) >= 1:
                        data["standard_sample"][color_type] = values[0]
                    if len(values) >= 2:
                        data["test_sample"][color_type] = values[1]
                    break
        print(data)
        return data
    except Exception as e:
        logger.error(f"Error in parse_extracted_data_safely: {str(e)}", exc_info=True)
        raise

def generate_html_table(data):
    """Generate HTML table from the data structure, with Delta E values only above the table"""
    html = f"""
    <h2>Color Data</h2>
    <p>Delta E2000: {data['delta_e_values'].get('Delta E2000', 'N/A')}</p>
    <p>Delta E76: {data['delta_e_values'].get('Delta E76', 'N/A')}</p>
    <table border="1">
        <thead>
            <tr>
                <th>Metric</th>
                <th>Standard Sample</th>
                <th>Test Sample</th>
            </tr>
        </thead>
        <tbody>
    """

    # Add rows for only the color metrics (not Delta E values)
    for metric in ["RGB", "CMYK", "HEX", "CIELAB", "LCH"]:
        standard_value = data['standard_sample'].get(metric, 'N/A')
        test_value = data['test_sample'].get(metric, 'N/A')
        html += f"""
            <tr>
                <td>{metric}</td>
                <td>{standard_value}</td>
                <td>{test_value}</td>
            </tr>
        """

    html += """
        </tbody>
    </table>
    """
    
    return html
@app.route('/upload', methods=['POST'])
def upload_image():
    logger.info("Starting file upload")
    temp_filepath = None
    cropped_filename = None
    
    try:
        # Basic file check
        if 'image' not in request.files:
            flash('No file selected')
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)

        # Validate file type
        if not allowed_file(file.filename):
            flash('Invalid file type')
            return redirect(request.url)

        # Temporary storage
        temp_filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(temp_filepath)
        logger.info(f"Temporary file saved to {temp_filepath}")

        # Process image with timeout and retrieve results
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(safe_process_image, temp_filepath)
            color_data, cropped_filename = future.result(timeout=MAX_PROCESSING_TIME)

        # Save sample data to the database
        sample_id = extract_sample_id(file.filename)
        safe_db_call(save_to_database, sample_id, color_data)

        # Verify cropped file and check if it’s accessible
        cropped_path = os.path.join(UPLOAD_FOLDER, cropped_filename)
        print(cropped_path)
        print(cropped_filename)
        if not os.path.exists(cropped_path):
            raise FileNotFoundError(f"Cropped image not found at {cropped_path}")

        return render_template(
            'display.html',
            sample_id=sample_id,
            color_data=color_data,
            delta_e_values=color_data.get('delta_e_values', {}),
            image_filename=cropped_filename
        )

    except TimeoutError:
        flash('Processing took too long. Please try again with a smaller image.')
        return redirect(request.url)
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        flash('An error occurred while processing the image')
        return redirect(request.url)
    finally:
        # Clean up temporary original file
        if temp_filepath and os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            logger.info(f"Cleaned up temporary file {temp_filepath}")





if __name__ == '__main__':
    # Configure logging to file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )
    app.run(debug=True)