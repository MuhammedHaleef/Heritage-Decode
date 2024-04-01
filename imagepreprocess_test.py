# import numpy as np
# from collections import Counter
# from itertools import product
#
#
# label_to_char = {
#         0: 'rock',
#         1: 's',
#         2: 'sh',
#         3: 'p',
#         4: 'ru2',
#         5: 'ru',
#         6: 'm',
#         7: 'k',
#         8: 'li',
#         9: 'dh',
#         10: 'pu',
#         11: 'th',
#         12: 'u',
#         13: 'h',
#         14: 'le',
#         15: 'Nhe',
#         16: 'ch',
#         17: 'thu',
#         18: 'b',
#         19: 'ri',
#         20: 'y',
#         21: 'shu',
#         22: 'r',
#         23: 'ki',
#         24: 'kadhi',
#         25: 'vi',
#         26: 'g',
#         27: 'thi',
#         28: 'n',
#         29: 'a',
#         30: 'dhi'
#     }
#
#
# def get_char_from_label(label):
#     # Assuming label_to_char dictionary is defined
#     return label_to_char.get(label, 'Unknown')  # Returns 'Unknown' if label not found
#
#
# most_repeated_labels = [0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 24, 24, 24, 24, 24, 24, 24, 24, 22, 22, 22, 6, 22, 22, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 24, 24, 24, 24, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8, 6, 6, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 29, 29, 29, 29, 29, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
# most_repeated_labels_array = np.array(most_repeated_labels)
# # Initialize an empty list to store new arrays without consecutive zeros
# new_arrays = []
#
# # Find indices where consecutive zeros occur
# zero_indices = np.where(np.diff(np.concatenate(([0], most_repeated_labels_array == 0, [0]))))[0]
#
# temp_zero_indices = [0]
# for zero_index in zero_indices:
#     temp_zero_indices.append(zero_index)
# temp_zero_indices.append(len(most_repeated_labels))
# zero_indices = temp_zero_indices
# # Iterate through the zero indices and split the array accordingly
# for start, end in zip(zero_indices[:-1:2], zero_indices[1::2]):
#     new_array = most_repeated_labels_array[start:end]
#     if len(new_array) > 0:
#         new_arrays.append(new_array)
#
# # # If the last group ends with zeros, add the remaining portion to new_arrays
# # if zero_indices[-1] == len(most_repeated_labels_array) - 1 and most_repeated_labels_array[-1] == 0:
# #     new_array = most_repeated_labels_array[zero_indices[-2] + 1:]
# #     if len(new_array) > 0:
# #         new_arrays.append(new_array)
#
# # Print the new arrays without consecutive zeros
# for idx, arr in enumerate(new_arrays):
#     print(f"Array {idx + 1}:", arr)
#
#
# selected_values = []
#
# # Step 1: Loop through each sliced array
# for arr in new_arrays:
#     # Step 2: Count the occurrences of each value in the sliced array
#     value_counts = Counter(arr)
#     temp_arr = []
#     temp_arr.extend([value for value, count in value_counts.items() if count > 10])
#     if len(temp_arr) > 0:
#         selected_values.append(temp_arr)
#     # # Step 3: Filter values with more than 5 occurrences and add to selected_values
#     # selected_values.extend([value for value, count in value_counts.items() if count > 5])
#
# # # Convert selected_values to a set to remove duplicates, then back to a list
# # selected_values = list(set(selected_values))
#
# # print("Selected values repeated more than 10 times:", selected_values)
#
#
# def convert_indices_to_chars(input_list):
#     # Initialize an empty list to store the converted characters
#     output_list = []
#
#     # Iterate through each sublist in the input list
#     for sublist in input_list:
#         # Map each index in the sublist to its corresponding character
#         chars = [label_to_char.get(index, 'Unknown') for index in sublist]
#         # Append the list of characters to the output list
#         output_list.append(chars)
#
#     return output_list
#
#
# # Convert indices to characters
# output_list = convert_indices_to_chars(selected_values)
#
# # Print the result
# for sublist in output_list:
#     print(sublist)
#
# # Use itertools.product() to generate all combinations
# combinations = list(product(*sublist))
#
# # Print the combinations
# for combination in combinations:
#     print(combination)
#

from collections import Counter
from itertools import product
import numpy as np


# Brahmi character and its respective label
label_to_char = {
    0: 'rock', 1: 's', 2: 'sh', 3: 'p', 4: 'ru2', 5: 'ru', 6: 'm', 7: 'k', 8: 'li', 9: 'dh', 10: 'pu', 11: 'th',
    12: 'u', 13: 'h', 14: 'le', 15: 'Nhe', 16: 'ch', 17: 'thu', 18: 'b', 19: 'ri', 20: 'y', 21: 'shu', 22: 'r',
    23: 'ki', 24: 'kadhi', 25: 'vi', 26: 'g', 27: 'thi', 28: 'n', 29: 'a', 30: 'dhi'
}

most_repeated_labels = [0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 24, 24, 24, 24, 24, 24, 24, 24, 22, 22, 22, 6, 22, 22, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 24, 24, 24, 24, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8, 6, 6, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 29, 29, 29, 29, 29, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]


# matches the respective char back from predicted label
def get_char_from_label(label):
    return label_to_char.get(label)


# Split array based on consecutive zeros / zero
def split_array_by_zeros(array):
    zero_indices = np.where(np.diff(np.concatenate(([0], array == 0, [0]))))[0]
    temp_zero_indices = [0]
    for zero_index in zero_indices:
        temp_zero_indices.append(zero_index)
    temp_zero_indices.append(len(array))
    zero_indices = temp_zero_indices
    return [array[start:end] for start, end in zip(zero_indices[:-1:2], zero_indices[1::2]) if len(array[start:end]) > 0]


# Get most repeated values with counts greater than 10
def get_selected_values(arrays):
    selected_values = []
    for arr in arrays:
        value_counts = Counter(arr)
        temp_arr = [value for value, count in value_counts.items() if count > 10]
        if temp_arr:
            selected_values.append(temp_arr)
    return selected_values


# Convert indices to characters
def convert_indices_to_chars(input_list):
    return [[get_char_from_label(index) for index in sublist] for sublist in input_list]


# Combine characters to form combinations
def get_combinations(characters):
    return list(product(*characters))


# Step 1: Split array by zeros
new_arrays = split_array_by_zeros(np.array(most_repeated_labels))

# Step 2: Get selected value slicesr of predicted array
selected_values = get_selected_values(new_arrays)

# Step 3: Convert indices to characters
output_list = convert_indices_to_chars(selected_values)

# Step 4: Generate combinations
combinations = get_combinations(output_list)

# Print combinations
for combination in combinations:
    print(combination)