import os
import cv2

def preprocess_images(input_folder, output_folder):

    # Iterate over each image file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Read the image
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            if img is not None:
                # Apply preprocessing steps
                img_processed = cv2.convertScaleAbs(img, alpha=-1, beta=1)  # Set exposure to -1
                img_processed = cv2.convertScaleAbs(img_processed, alpha=-1, beta=50)  # Set contrast to 60
                # img_processed = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

                # Save the preprocessed image to the output folder
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, img_processed)

                print(f"Processed: {filename}")

# Example usage:
input_folder = "C:/Users\MuhammedHaleef\OneDrive\Documents\AI & DS/2nd Year\CM2603 DSGP\Final Project\Labelling Dataset\Complete Dataset/Unprocessed"
output_folder = "C:/Users\MuhammedHaleef\OneDrive\Documents\AI & DS/2nd Year\CM2603 DSGP\Final Project\Labelling Dataset\Complete Dataset\Preprocessed"

preprocess_images(input_folder, output_folder)
