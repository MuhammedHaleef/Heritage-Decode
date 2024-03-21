import cv2
import numpy as np
from collections import Counter

character_color_labels = {
    's': '#ff000f',
    'sh': '#650006',
    'p': '#0f00ff',
    'ru': '#070348',
    'ru-2': '#6713ec',
    'm': '#5e5c80',
    'm2': '#8c89ce',
    'k': '#8c4747',
    'li': '#ff03a1',
    'dh': '#0ffe00',
    'pu': '#7f7ce7',
    'th': '#ff9595',
    'u': '#681d67',
    'h': '#f7e500',
    'le': '#00f4ff',
    'Nhe': '#6e6c93',
    'ch': '#3b8b9c',
    'thu': '#14593b',
    'b': '#fe8100',
    'ri': '#4fa681',
    'y': '#b7e29f',
    'shu': '#dd4d4d',
    'r': '#aa6eff',
    'ki': '#b81dba',
    'kadhi': '#835812',
    'v': '#7ca419',
    'g': '#517885',
    'thi': '#898989',
    'n': '#6d0000',
    'a': '#718693',
    'dhi': '#05ff92',
    'chi': '#f0cc13'}
# print(character_color_labels.values())


def most_common_color_in_column(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path,  cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = img.shape

    # Initialize a list to store most common colors for each column
    most_common_characters = []

    # Define the color to ignore
    ignore_color = (59, 65, 57)  # RGB value for hex #3b4139

    # Iterate through each column
    for x in range(img_width):
        column_colors = []
        # Iterate through each pixel in the column
        for y in range(img_height):
            pixel = img[y, x]
            pixel_color = rgb_to_hex(tuple(pixel))  # Convert pixel color to hex

            # print(pixel_color in character_color_labels.values())
            if tuple(pixel) != ignore_color and pixel_color in character_color_labels.values():
                column_colors.append(tuple(pixel))

        # Count the occurrences of each color in the column
        color_counts = Counter(column_colors)

        # Check if the column has only the ignore color
        if len(color_counts) == 0:
            most_common_color = ignore_color
        else:
            # Get the most common color in the column (excluding ignore color)
            most_common_color = color_counts.most_common(1)[0][0]

        # most_common_colors.append(most_common_color)
        # Find the corresponding character for the most common color
        most_common_character = next((key for key, value in character_color_labels.items() if value == most_common_color), None)
        most_common_characters.append(most_common_character)

    return most_common_characters


# Function to convert RGB color tuple to hex string
def rgb_to_hex(rgb_color):
    return '#{:02x}{:02x}{:02x}'.format(*rgb_color)


# Convert hex color code to RGB tuple
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


# Function to find the closest color value in the dictionary
def find_closest_color_label(rgb_color):
    closest_label = None
    min_distance = float('inf')
    for label, hex_color in character_color_labels.items():
        color_rgb = hex_to_rgb(hex_color)
        distance = np.linalg.norm(np.array(rgb_color) - np.array(color_rgb))
        if distance < min_distance:
            min_distance = distance
            closest_label = label
    return closest_label


# Example usage:
image_path = "images/prediction.jpg"  # Replace with your image path
most_common_characters = most_common_color_in_column(image_path)
# most_common_colors_hex = [rgb_to_hex(color) for color in most_common_colors]
print("Most common colors in each column:", most_common_characters)



