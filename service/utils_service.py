import json
import os


def check_and_update_images(path_to_images, path_to_json):
    # Get list of image file names in directory
    image_files = os.listdir(path_to_images)

    # Load existing JSON list
    with open(path_to_json, 'r') as json_file:
        existing_images = json.load(json_file)

    # Compare the two lists
    if set(image_files) != set(existing_images):
        # If there's a difference, overwrite the JSON
        with open(path_to_json, 'w') as json_file:
            json.dump(image_files, json_file)
        return True

    # If there's no difference, do nothing and return False
    return False
