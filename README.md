# Geo Orthophoto Fetcher

A Python tool for fetching and processing orthophoto images based on geographic coordinates.

## Features

- Convert between WGS84 (EPSG:4326) and UTM (EPSG:25832) coordinate systems
- Fetch required image files based on geographic location and radius
- Process multiple image files concurrently
- Combine multiple image files into a single composite image
- Resize output images to a standard size (256x256 pixels)

## Important Note on File Naming Convention

The filenames in this dataset encode the geospatial coordinates range. For example:

`dop10rgbi_32_280_5659.jp2`

This filename corresponds to:
- Latitude ranging from 280,000 to 280,999
- Longitude ranging from 5,659,000 to 5,659,999

These coordinates are in the EPSG:25832 CRS (Coordinate Reference System).

### Image Structure

The orthophoto files are expected to have 4 channels:
1. Red
2. Green
3. Blue
4. Additional channel (e.g., near-infrared)

## Requirements

- Python 3.6+
- numpy
- rasterio
- opencv-python (cv2)
- pyproj
