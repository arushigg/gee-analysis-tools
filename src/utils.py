import ee
import os
from datetime import datetime
import geemap
from typing import Union
import logging
import io
import contextlib
import re
import math


def auth():
    """
    Either authenticate with saved credentials in .config/earthengine/ or create a popup to
    authenticate.
    """
    if os.path.exists(ee.oauth.get_credentials_path()) is False:
        ee.Authenticate()
    else:
        print('Found your authentication credentials.')
        ee.Initialize(project="ap-satellite-imagery")

def to_ee_date(date: Union[ee.Date, datetime, str]) -> ee.Date:
    """
    Convert a date input into a ee.Date. 
    :param date: A date of format ee.Date, datetime object, or YYYY-MM-DD string.
    :returns: The equivalent ee.Date.
    """
    if type(date) == ee.Date:
        return date
    elif type(date) == datetime:
        return ee.Date(date.strftime("%Y-%m-%d"))
    elif type(date) == str:
        try:
            return ee.Date(date)
        except:
            logging.error(f"Could not convert input {date} to ee.Date")
    else:
        logging.error(f"Invalid type for input {date}.")

def shp_to_ee_poly(shp_path: str) -> ee.Geometry:
    """
    Convert a shapefile to an ee FeatureCollection.
    :param shp_path: Path to the shapefile to load.
    :returns: An ee.Geometry for that shapefile.
    """
    gdf = geemap.shp_to_ee(shp_path)
    fc = geemap.geopandas_to_ee(gdf)
    return fc

def get_zoom_level(
        bounds: ee.Geometry, map_width: int = 800, map_height: int = 600
    ) -> int:
    """
    Find the zoom level dynamically, based on the geometry to show and window in which to 
    display the map.
    :param bounds: The geometry to zoom to.
    :param map_width: The width in pixels of the map display.
    :param map_height: The height in pixels of the map display.
    :returns: The appropriate zoom level.
    """
    # From Chat GPT
    coords = bounds.bounds().coordinates().getInfo()[0]
    lon_delta = coords[2][0] - coords[0][0]
    lat_delta = coords[2][1] - coords[0][1]

    # Calculate the map's scale in meters per pixel
    lon_scale = 111320 * (lon_delta / map_width)
    lat_scale = 111320 * (lat_delta / map_height)

    # Determine the zoom level based on the scale
    zoom_level = min(int(10 - (lon_scale + lat_scale) / 2000), 20)
    return zoom_level

def get_scale(img: Union[ee.Image, ee.ImageCollection]) -> int:
    """
    Find the scale (or, pixel size) of an image or image collection.
    :param img: The image to check.
    :returns: The scale in meters.
    """
    if type(img) == ee.Image:
        return img.select(0).projection().nominalScale().getInfo()
    elif type(img) == ee.ImageCollection:
        return img.first().select(0).projection().nominalScale().getInfo()
    else:
        logging.error(f"Must pass input of type Image or ImageCollection, not {type(img)}")


def get_bands(img: Union[ee.Image, ee.ImageCollection]) -> list[str]:
    """
    Get a list of the bands in an image of image collection.
    :param img: The image to check.
    :returns: A list of band names.
    """
    if type(img) == ee.Image:
        return img.bandNames().getInfo()
    elif type(img) == ee.ImageCollection:
        return img.first().bandNames().getInfo()
    else:
        logging.error(f"Must pass input of type Image or ImageCollection, not {type(img)}")


def is_date_earlier(date1: ee.Date, date2: ee.Date) -> bool:
    """
    Checks whether one ee.Date is before another.
    :param date1: The first date.
    :param date2: The second date.
    :returns: True if date1 is before date2.
    """
    return date1.millis().lt(date2.millis()).getInfo()

def format_date(date: ee.Date) -> str:
    """
    Format an ee.Date for printing (eg. Jun. 25, 2024).
    :param date: The date to format.
    :returns: A formatted string.
    """
    return date.format('MMM. dd, YYYY').getInfo()

def get_image_date(img: ee.Image) -> str:
    """
    Get the date of an image as a formatted string.
    :param img: The image to date.
    :returns: A formatted string.
    """
    timestamp = img.get('system:time_start')
    date = ee.Date(timestamp)
    return format_date(date)

def get_latest_image(imgs: ee.ImageCollection) -> ee.Image:
    """
    Get the latest image in a collection.
    :param imgs: The collection.
    :returns: The last image in the collection.
    """
    return imgs.sort("system:time_start", False).first()

def export_image(img: ee.Image, dest: str, region: ee.Geometry, scale: int = None) -> None:
    """
    This doesn't work when the image has undergone reprocessing to be visualized and I can't 
    figure out why. I think it's something to do with this function that's called by
    geemap.ee_export_image: https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl.
    In case it's helpful, here's geemap.ee_export_image: https://github.com/gee-community/geemap/blob/master/geemap/common.py#L160
    """
    image = img.clip(region)

    if scale is None:
        scale = 10
    
    # the function doesn't throw an error, so we capture its printing and parse that
    # equivalent to a try-catch block
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        geemap.ee_export_image(
            image, filename=dest, scale=scale, region=region, file_per_band=False
        )
    output = f.getvalue()

    if "error" in output:
        actual_bytes, desired_bytes = re.findall(r"([0-9]*) bytes", output)
        scale_factor = (int(actual_bytes) / int(desired_bytes)) ** 0.5
        new_scale = math.ceil(scale * scale_factor) + 1 # just to be safe
        logging.warning(f"Actual scale of {scale} was too small, changed to {new_scale} to stay within size limits.")
        geemap.ee_export_image(
            image, filename=dest, scale=new_scale, region=region, file_per_band=False
        )

def get_country_filter(country_code: int):
    """
    Codes are here: https://www.fao.org/in-action/countrystat/news-and-events/events/training-material/gaul-codes2014/en/
    """
    countries = ee.FeatureCollection('FAO/GAUL/2015/level0')
    country = countries.filter(ee.Filter.eq('ADM0_CODE', country_code))
    country_geom = country.geometry()
    return ee.Filter.bounds(country_geom)