import ee
from datetime import datetime
import logging
import src.utils as utils
import geemap


class LandSat:
    """
    Wrapper for Landsat imagery (Collection 2, Tier 1 + real time). Main added utility
    is producing a minimally-cloudy composite for the given time and geography.
    """

    start_date: ee.Date
    end_date: ee.Date
    geometry: ee.Geometry
    collection: ee.ImageCollection
    map: geemap.Map
    # map of label to layer
    layers: dict[str, ee.Image]

    L8_SR_ID = 'LANDSAT/LC08/C02/T1_RT'
    LS_VIS_PARAMS = {'bands': ['SR_B4', 'SR_B3', 'SR_B2'], "gamma": 1.4}

    def __init__(
        self,
        geometry: ee.Geometry,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> None:
        """
        Initialize with the desired geometry and date range.
        :param geometry: A point or polygon to focus on.
        :param start_date: The first date of interest, Jun. 27, 2015 by default.
        :param end_date: The last date of interest, today by default.
        """
        # Only need to project when taking the composite, but store the projection
        # just in case. Scale of Dynamic World is 10m.
        self.PROJECTION = ee.Projection("EPSG:3857").atScale(10)
        # The first date data is available
        self.FIRST_DATE = ee.Date("2015-06-27")

        self.geometry = geometry
        self.start_date = utils.to_ee_date(start_date)
        self.end_date = utils.to_ee_date(end_date)
        self._load_collection()

        self.reset_map()


    def get_filter(self) -> ee.Filter:
        """
        Creates a filter corresponding to the instance's date range and geometry.
        :returns: A filter object.
        """
        loc_filter = ee.Filter.bounds(self.geometry)
        time_filter = ee.Filter.date(self.start_date, self.end_date)
        return ee.Filter.And(loc_filter, time_filter)
    
    def get_num_images(self) -> int:
        """
        :returns: The number of images in the collection.
        """
        return self.collection.size().getInfo()

    def _load_collection(self):
        """
        Load the Landsat 8 imagery corresponding to the instance's date range and geometry.
        :raises Exception: if there are no images in the collection.
        """
        self.collection = (
            ee.ImageCollection(self.L8_SR_ID)
            .filter(self.get_filter())
            .sort("system:time_start")
        )
        if self.get_num_images() == 0:
            logging.error(
                "The filters are too strict, try changing the date range or geometry."
            )

    def reset_map(self):
        """
        Clear the layers on the map, recenter on the instance geometry, and apply formatting.
        """
        self.map = geemap.Map()
        self.layers = {}
        self.map.center_object(self.geometry)
        # Arbitrarily defaults to zoom level 12 if the geometry is a point
        if type(self.geometry) == ee.Geometry.Point:
            self.map.center_object(zoom=12)
        self.map.add_layer_control()

    @staticmethod
    def mask_clouds(image: ee.Image) -> ee.Image:
        """
        Mask the cloudy pixels and return the image.
        :param image: The image to mask.
        :returns: The same image with cloudy pixels masked.
        """
        qa = image.select('QA_PIXEL')
        # Bitwise extract cloud bit (4)
        cloud_mask = (
            qa.bitwiseAnd(1 << 3).eq(0)
            .And(qa.bitwiseAnd(1 << 5).eq(0))
        )
        return image.updateMask(cloud_mask)

    def get_composite_image(self, clip=False) -> ee.Image:
        """
        Aggregate all the images into a single composite.
        :param clip: Whether to clip the composite to the specified geometry (or
          return the maximal geometry available).
        :returns: The composite image.
        """
        comp_img = (
            self.collection
            .map(LandSat.mask_clouds)
            .sort('CLOUD_COVER')
            .mosaic()
        )
        if clip:
            comp_img = comp_img.clip(self.geometry)
        return comp_img
    
    def add_image_to_map(self, image: ee.Image, label: str) -> None:
        """
        Add an image to the map with the given label.
        :param image: The image to map.
        :param label: The label for the layer.
        """
        self.map.add_layer(image, self.LS_VIS_PARAMS, label)
        self.layers[label] = image