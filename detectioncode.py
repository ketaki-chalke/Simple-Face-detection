import cv2
from deepface import DeepFace
from tkinter import Tk, filedialog
import os

def select_image():
    Tk().withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.webp")])
    return file_path

# Load reference images and their names from a directory
def load_reference_images(directory):
    reference_images = []
    reference_image_names = []
    for filename in os.listdir(directory):
        if filename.endswith((".webp", ".jpg", ".jpeg", ".png")):
            img_path = os.path.join(directory, filename)
            reference_images.append(img_path)
            reference_image_names.append(os.path.splitext(filename)[0])  # Store the name without extension
    return reference_images, reference_image_names

try:
    reference_images_directory = "C:\\Users\\230941\\Documents\\faces"  # Replace with the path to your database of images
    reference_images, reference_image_names = load_reference_images(reference_images_directory)

    # Allow the user to select an image
    image_path = select_image()

    if image_path:
        frame = cv2.imread(image_path)
        if frame is not None:
            # Iterate through each reference image in the database
            for reference_img_path, reference_name in zip(reference_images, reference_image_names):
                # Use Facenet model for better accuracy
                result = DeepFace.verify(frame, reference_img_path, model_name='Facenet', enforce_detection=False)
                if result['verified']:
                    print(f"Matched: {reference_name} with confidence: {result['distance']}")
                    break  # Break the loop if a match is found
            else:
                print("No match found in the reference images.")
        else:
            print("Error reading the image.")
    else:
        print("No image selected.")
except Exception as e:
    print(f"An error occurred: {e}")
