import concurrent.futures
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window


IMAGE_DIR = "your-image-directory"
FILE_NAME_FORMAT = "dop10rgbi_32_{lon}_{lat}.jp2"

# Transformers for coordinate conversion
transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
transformer_to_geo = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)


def calculate_coordinate_bounds(lat_utm, lon_utm, radius):
    """
    Calculate the minimum and maximum UTM coordinates based on the provided UTM coordinates and radius.
    :param lat_utm: Latitude in UTM coordinates
    :param lon_utm: Longitude in UTM coordinates
    :param radius: Radius in meters
    :return: Tuple containing min and max latitudes and longitudes
    """
    lat_min = lat_utm - radius
    lat_max = lat_utm + radius
    lon_min = lon_utm - radius
    lon_max = lon_utm + radius
    return lat_min, lat_max, lon_min, lon_max


def get_required_images(lat_utm, lon_utm, radius):
    """
    Generate a list of required image filenames based on UTM coordinates and radius.
    :param lat_utm: Latitude in UTM coordinates
    :param lon_utm: Longitude in UTM coordinates
    :param radius: Radius in meters
    :return: List of filenames for the required images
    """
    lat_min, lat_max, lon_min, lon_max = calculate_coordinate_bounds(lat_utm, lon_utm, radius)

    # Adjust coordinates for filename formatting
    lat_min = math.floor(lat_min) // 1000
    lat_max = math.ceil(lat_max) // 1000
    lon_min = math.floor(lon_min) // 1000
    lon_max = math.ceil(lon_max) // 1000

    required_images = []
    for lon in range(lon_min, lon_max + 1):
        for lat in range(lat_min, lat_max + 1):
            filename = filename = FILE_NAME_FORMAT.format(lon=lon, lat=lat)
            required_images.append(filename)

    return required_images


def parse_filename(filename):
    """
    Parse latitude and longitude from the filename.
    :param filename: Filename of the image
    :return: Tuple containing longitude and latitude
    """
    parts = filename.split('_')
    lon = int(parts[2])
    lat = int(parts[3])
    return lon, lat


def identify_position(filenames):
    """
    Identify the positions of the images based on their filenames.
    :param filenames: List of filenames
    :return: Dictionary mapping positions to filenames
    """
    if len(filenames) == 1:
        return {'single': filenames[0]}
    elif len(filenames) not in [2, 4]:
        raise ValueError("Function expects either 1, 2, or 4 filenames.")

    parsed_files = [{'file': fname, 'lon': parse_filename(fname)[0], 'lat': parse_filename(fname)[1]} for fname in filenames]

    if len(filenames) == 2:
        file1, file2 = parsed_files

        if file1['lon'] == file2['lon']:  # Same longitude, different latitude
            if file1['lat'] < file2['lat']:
                return {'bottom': file1['file'], 'top': file2['file']}
            else:
                return {'bottom': file2['file'], 'top': file1['file']}
        elif file1['lat'] == file2['lat']:  # Same latitude, different longitude
            if file1['lon'] < file2['lon']:
                return {'left': file1['file'], 'right': file2['file']}
            else:
                return {'left': file2['file'], 'right': file1['file']}
        else:
            raise ValueError("The two files are neither horizontally nor vertically adjacent.")

    elif len(filenames) == 4:
        parsed_files.sort(key=lambda x: (x['lon'], x['lat']))

        return {
            'bottom_left': parsed_files[0]['file'],
            'top_left': parsed_files[1]['file'],
            'bottom_right': parsed_files[2]['file'],
            'top_right': parsed_files[3]['file']
        }

def read_partial_image(file_name, lat_utm, lon_utm, radius, image_dir):
    """
    Read a partial image based on UTM coordinates and radius.
    :param file_name: Filename of the image
    :param lat_utm: Latitude in UTM coordinates
    :param lon_utm: Longitude in UTM coordinates
    :param radius: Radius in meters
    :param image_dir: Directory where the image files are located
    :return: Tuple containing the filename and the image data
    """
    lat_min, lat_max, lon_min, lon_max = calculate_coordinate_bounds(lat_utm, lon_utm, radius)

    try:
        with rasterio.open(os.path.join(image_dir, file_name)) as src:
            # Calculate pixel coordinates
            transform = src.transform
            col_start, row_start = ~transform * (lon_min, lat_max)
            col_stop, row_stop = ~transform * (lon_max, lat_min)

            row_start, col_start = int(row_start), int(col_start)
            row_stop, col_stop = int(row_stop), int(col_stop)

            window = Window(col_start, row_start, col_stop - col_start, row_stop - row_start)

            data = src.read(window=window)
    except rasterio.errors.RasterioIOError:
        data = None

    return file_name, data


def build_complete_image(image_dict, radius):
    """
    Build a complete image from pieces according to their position, handling different image sizes.
    :param image_dict: Dictionary with filename as key and image array as value
    :return: Complete image as a numpy array
    """
    num_images = len(image_dict)

    if num_images not in [1, 2, 4]:
        raise ValueError("The dictionary should contain 1, 2, or 4 images.")

    # Identify positions of the images
    positions = identify_position(list(image_dict.keys()))

    # Create an empty image
    # Proportion of distance (1 km) to number of pixels per side (10000) is equal to 1/10
    side_length = radius * 10 * 2

    # Image with 4 channels
    complete_image = np.zeros((4, side_length, side_length), dtype=np.int16)

    if num_images == 1:
        # If there's only one image, return it
        img = next(img for img in image_dict.values())
        if img is not None:
            return img
        else:
            return complete_image

    elif num_images == 2:
        # If there are two images, place them according to their identified positions
        if 'left' in positions and 'right' in positions:
            left_img = image_dict[positions['left']]
            right_img = image_dict[positions['right']]
            if left_img is not None:
                complete_image[:, :left_img.shape[1], :left_img.shape[2]] = left_img
            if right_img is not None:
                complete_image[:, :right_img.shape[1], -right_img.shape[2]:] = right_img
        elif 'top' in positions and 'bottom' in positions:
            top_img = image_dict[positions['top']]
            bottom_img = image_dict[positions['bottom']]
            if top_img is not None:
                complete_image[:, :top_img.shape[1], :top_img.shape[2]] = top_img
            if bottom_img is not None:
                complete_image[:, -bottom_img.shape[1]:, :bottom_img.shape[2]] = bottom_img

    elif num_images == 4:
        # If there are four images, place them in a 2x2 grid with correct positioning
        for pos in ['bottom_left', 'top_left', 'bottom_right', 'top_right']:
            img = image_dict[positions[pos]]
            if img is None:
                continue

            h, w = img.shape[1], img.shape[2]
            if pos == 'bottom_left':
                complete_image[:, -h:, :w] = img
            elif pos == 'top_left':
                complete_image[:, :h, :w] = img
            elif pos == 'bottom_right':
                complete_image[:, -h:, -w:] = img
            elif pos == 'top_right':
                complete_image[:, :h, -w:] = img

    return complete_image


def process_images(lat_utm, lon_utm, radius, image_dir):
    """
    Process all required images based on UTM coordinates and radius.
    :param lat_utm: Latitude in UTM coordinates
    :param lon_utm: Longitude in UTM coordinates
    :param radius: Radius in meters
    :param image_dir: Directory where the image files are located
    :return: Dictionary with filenames as keys and image arrays as values
    """
    required_images = get_required_images(lat_utm, lon_utm, radius)

    # Check if there's only one image to process
    if len(required_images) == 1:
        # Process the single image synchronously
        result = [read_partial_image(required_images[0], lat_utm, lon_utm, radius, image_dir)]
    else:
        # Use ThreadPoolExecutor for multiple images
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(read_partial_image, file_name, lat_utm, lon_utm, radius, image_dir)
                for file_name in required_images
            ]
            result = [future.result() for future in concurrent.futures.as_completed(futures)]

    return dict(result)


def get_image(lat, long, radius):
    """
    Fetch and process images based on latitude, longitude, and radius.
    :param lat: Latitude in WGS84 (EPSG:4326)
    :param lon: Longitude in WGS84 (EPSG:4326)
    :param radius: Radius in meters (up to 100)
    :return: Processed image as a numpy array (256x256 pixels)
    """
    if radius > 100:
        raise ValueError("Radius should not exceed 100 meters.")

    # Convert lat/long to UTM
    lon_utm, lat_utm = transformer_to_utm.transform(long, lat)

    # Process images
    processed_images = process_images(lat_utm, lon_utm, radius, IMAGE_DIR)

    # Build complete image
    complete_image = build_complete_image(processed_images, radius)

    # Resize the image to 256x256
    resized_image = cv2.resize(complete_image[:3].transpose((1, 2, 0)), (256, 256), interpolation=cv2.INTER_LINEAR)

    return resized_image


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example coordinates (replace with your desired coordinates)
    lat = 51.5074  # London latitude
    lon = -0.1278  # London longitude
    radius = 100   # meters

    # Fetch the image
    image = get_image(lat, lon, radius)

    # Display the image
    plt.imshow(image)
    plt.title(f"Orthophoto at Lat: {lat}, Lon: {lon}, Radius: {radius}m")
    plt.axis('off')
    plt.show()

    print(f"Image shape: {image.shape}")
    print(f"Image data type: {image.dtype}")