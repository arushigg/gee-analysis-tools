# gee-satellite-imagery

## Project goal

This project started at Associated Press in order to provide tools that ease exploration of satellite imagery through Google Earth Engine. The intention isn't to replicate the breadth of functionality Google Earth Engine provides, but to add a layer of abstraction so reporters (and others!) can assess whether an idea has legs with less overhead. This project focuses on the [Dynamic World](https://dynamicworld.app/) dataset, which provides near real-time classification of land cover, making it most useful for identifying changes in land cover like deforestation, flooding, or urban expansion.

## Project notes

### Getting started
If you have a credentials file, upload it to the `.config/earthengine/` folder. Calling `utils.auth()` will load these credentials, if they exist, or trigger a pop up window to authenticate your account. For more details, see Google's instructions on authenticating your Earth Engine account [here](https://developers.google.com/earth-engine/guides/auth).

Use `python .first_install.py` to install packages and create the virtual environment.

### `dynamic_world.py`

This file defines a wrapper class to condense images with different geographies over a chosen time period into a smaller number of composites. Each composite, or individual image, can be visualized as a map of land cover type, a Sentinel 2 image, or a bar chart showing the share of each land cover type. The example notebook `analysis/dw_example.ipynb` shows how to use the class to explore an area.

### `landsat.py`

The Landsat tools are relatively barebones -- the wrapper class loads a collection from a certain time and geography to create a single composite that can be exported. The class simplifies masking clouds, creating mosaics, and exporting at the appropriate scale.

### `utils.py`

Additional functions to convert between `ee` data types and more universal file types, extract common metadata from images, and set scale dynamically.

---
*Thank you to the data team at AP, particularly MK Wildeman, for shaping this project!*
