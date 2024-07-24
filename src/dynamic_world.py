import ee
from datetime import datetime
import logging
import src.utils as utils
import geemap
import math
import pandas as pd
import altair as alt
import os


class DynamicWorld:
    """
    Wrapper for a map of Dynamic World imagery. Each instance has an associated geographic and temporal
    focus, and stores different layers visualizing that data on a single map.

    Dynamic World images have a probability from 0 to 1 for each of 9 land cover types (listed in BANDS)
    and a 'label' band that stores the value of the highest probability band.
    """

    start_date: ee.Date
    end_date: ee.Date
    geometry: ee.Geometry
    collection: ee.ImageCollection
    map: geemap.Map
    layers: dict[str, ee.Image]

    # IDs to load imagery from GEE
    ID = "GOOGLE/DYNAMICWORLD/V1"
    SENTINEL_ID = "COPERNICUS/S2_HARMONIZED"
    BANDS = [
        "water",
        "trees",
        "grass",
        "flooded_vegetation",
        "crops",
        "shrub_and_scrub",
        "built",
        "bare",
        "snow_and_ice",
    ]
    COLORS = [
        "#419BDF",
        "#397D49",
        "#88B053",
        "#7A87C6",
        "#E49635",
        "#DFC35A",
        "#C4281B",
        "#A59B8F",
        "#B39FE1",
    ]
    BAND_TO_COLOR = dict(zip(BANDS, COLORS))
    # Parameters to map Dynamic World layers
    DW_VIS_PARAMS = {"min": 0, "max": 0.8}
    # Parameters to map Sentinel 2 layers
    S2_VIS_PARAMS = {"min": 0, "max": 3000, "bands": ["B4", "B3", "B2"]}
    # Set in __init__ after EE is initialized
    PROJECTION: ee.Projection
    FIRST_DATE: ee.Date

    # Format to label layers of the map
    COMPOSITE_LABEL = "Composite: {} to {}"  # start date, end date
    IMAGE_LABEL = (
        "Image {} ({}): {}"  # image #, date, imagery source (Dynamic World or Sentinel 2)
    )
    SINGLE_BAND_LABEL = "Prob. of {} > {:.3f}" # band name, threshold

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
        self.start_date = None
        self.end_date = None
        self.set_date_range(start_date, end_date)

        self.reset_map()

    def set_date_range(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> None:
        """
        Set the date range for the instance, repulling imagery accordingly. Chooses the
        maximum range of Jun. 27, 2015 until the current date by default.
        :param start_date: The first date of interest, Jun. 27, 2015 by default.
        :param end_date: The last date of interest, today by default.
        """

        if start_date is not None:
            self.start_date = utils.to_ee_date(start_date)
            if utils.is_date_earlier(self.start_date, self.FIRST_DATE):
                logging.warning(
                    "Dynamic World does not have imagery before Jun. 27, 2015... setting start date to Jun. 27, 2015."
                )
                self.start_date = self.FIRST_DATE
        else:
            if self.start_date is None:
                self.start_date = self.FIRST_DATE

        if end_date is not None:
            self.end_date = utils.to_ee_date(end_date)
            if utils.is_date_earlier(self.end_date, self.FIRST_DATE):
                logging.warning(
                    "Dynamic World does not have imagery before Jun. 27, 2015... setting end date to Jun. 27, 2015."
                )
                self.end_date = self.FIRST_DATE
        else:
            if self.end_date is None:
                self.end_date = utils.to_ee_date(datetime.now())

        if utils.is_date_earlier(self.end_date, self.start_date):
            logging.error(
                "End date is earlier than start date. Try setting the date range again."
            )

        self._load_collection()

    def set_geometry(self, geometry: ee.Geometry) -> None:
        """
        Set the geometry for the instance, repulling imagery accordingly.
        :param geometry: A point or polygon to focus on.
        """
        self.geometry = geometry
        self._load_collection()

    def get_filter(self) -> ee.Filter:
        """
        Creates a filter corresponding to the instance's date range and geometry.
        :returns: A filter object.
        """
        loc_filter = ee.Filter.bounds(self.geometry)
        time_filter = ee.Filter.date(self.start_date, self.end_date)
        return ee.Filter.And(loc_filter, time_filter)

    def _load_collection(self):
        """
        Load the Dynamic World imagery corresponding to the instance's date range and geometry.
        :raises Exception: if there are no images in the collection.
        """
        self.collection = (
            ee.ImageCollection(self.ID)
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
        self.map.add_legend(legend_dict=self.BAND_TO_COLOR)

    def get_bands(self) -> list[str]:
        """
        :returns: A list of the bands in the collection.
        """
        return utils.get_bands(self.collection)

    def get_num_images(self) -> int:
        """
        :returns: The number of images in the collection.
        """
        return self.collection.size().getInfo()

    def get_earliest_image(self) -> ee.Image:
        """
        :returns: The first image in the collection.
        """
        return self.collection.sort("system:time_start").first()

    def get_latest_image(self) -> ee.Date:
        """
        :returns: The latest image in the collection.
        """
        return utils.get_latest_image(self.collection)

    def get_image_at_index(self, index: int) -> ee.Image:
        """
        :returns: The image at a particular index in the collection.
        """
        img_list = self.get_collection_as_list()
        return img_list[index]

    def get_composite_image(self, imgs: ee.ImageCollection = None) -> ee.Image:
        """
        Aggregates all the images in a collection into a single image, taking the mode
        for the label band and the mean of the probabilities for other bands. Since a single image
        may be missing information due to cloud cover or just smaller geography, composites
        can provide a fuller picture of an area.
        :param imgs: The collection to compose, the instance's entire image collection by default.
        """
        if imgs is None:
            imgs = self.collection

        label_band = imgs.select("label").reduce(ee.Reducer.mode()).rename("label")
        prob_bands = (
            imgs.select(self.BANDS).reduce(ee.Reducer.mean()).rename(self.BANDS)
        )
        # Combine the label and probability bands & project -- EE seems to get confused if we don't
        # set a projection for composites
        all_bands = prob_bands.addBands(label_band).setDefaultProjection(
            self.PROJECTION
        )

        return all_bands

    def get_all_layers(self) -> dict[str, ee.Image]:
        return self.layers
    
    def get_layer_by_name(self, name: str) -> ee.Image:
        image = self.layers.get(name)
        if image is None:
            logging.error(f"No layer with the name '{name}'. Valid names are: {', '.join(self.layers.keys())}")
        return image

    def _get_dw_s2_collection(self) -> ee.Join:
        """
        Create a collection that has each Dynamic World image linked to the source
        Sentinel 2 image.
        :returns: An image collection with the image pairs.
        """
        # Filter the Sentinel 2 imagery to the same date range and geography
        s2 = ee.ImageCollection(self.SENTINEL_ID).filter(self.get_filter())
        # Join on the image index
        dw_s2 = ee.Join.saveFirst("s2_img").apply(
            self.collection,
            s2,
            ee.Filter.equals(leftField="system:index", rightField="system:index"),
        )
        return dw_s2

    def get_collection_as_list(self) -> list[ee.Image]:
        """
        :returns: The image collection as a list of images.
        """
        as_ee_list = self.collection.toList(self.collection.size())
        as_python_list = [
            ee.Image(as_ee_list.get(i)) for i in range(self.get_num_images())
        ]
        return as_python_list

    def bucket_by_time(self, num_buckets: int = 5) -> list[ee.ImageCollection]:
        """
        Divide the collection into time periods and return a collection for each window. If the
        number of images doesn't divide evenly into the desired number of buckets, there will
        be overlap between the buckets (ie. the same image will appear in multiple collections) 
        instead of buckets being different sizes.
        :param num_buckets: The number of time periods to segment into.
        :returns: A list of image collections, one for each bucket.
        """
        images_per_layer = math.ceil(self.get_num_images() / num_buckets)
        logging.info(
            f"There are {self.get_num_images()} images and {images_per_layer} images in each layer."
        )
        dw_list = self.get_collection_as_list()
        collecs = []
        for i in range(num_buckets):
            # Since we round after multiplying by i, the start index can be less than 
            # i * images_per_layer, allowing for overlap between buckets.
            start_i = round(i * self.get_num_images() / num_buckets)
            last_i = min(self.get_num_images(), start_i + images_per_layer)
            collecs.append(ee.ImageCollection(dw_list[start_i:last_i]))
        return collecs

    def visualize_image(self, img: ee.Image) -> ee.Image:
        """
        Transform a 10-band Dynamic World image into a 3-band RGB image that can be mapped. The
        map is colored by the most probable land cover type and the elevation corresponds to the
        probability of that type.
        :returns: An RGB image.
        """
        # Set the color based on the 'label' band
        rgb_img = (
            img.select("label").visualize(min=0, max=8, palette=self.COLORS).divide(255)
        )
        # Get the actual probability of that band
        top_prob = img.select(self.BANDS).reduce(ee.Reducer.max()).multiply(100)
        # Use the hillshade visualization to combine the color and elevation
        top_prob_hillshade = ee.Terrain.hillshade(top_prob).divide(255)
        return rgb_img.multiply(top_prob_hillshade)

    def add_layer(self, img: ee.Image, label: str, viz_params: dict = None, is_dw: bool = True) -> None:
        """
        Wrapper to add a layer to the map. Also updates the internal list of images.
        """
        if is_dw:
            viz_img = self.visualize_image(img)
            if viz_params is not None:
                self.map.add_layer(viz_img, viz_params, label)
            else:
                self.map.add_layer(viz_img, self.DW_VIS_PARAMS, label)
            self.layers[label + "_raw"] = img
            self.layers[label] = viz_img
        else:
            if viz_params is not None:
                self.map.add_layer(img, viz_params, label)
            else: 
                self.map.add_layer(img, {}, label)
            self.layers[label] = img

    def map_image_at_index(self, index: int = 0) -> None:
        """
        Map the Dynamic World land cover and the corresponding Sentinel 2 source for the image
        at a particular index.
        :param index: The index of the image to map, 0 by default. The index can be negative, but
            its absolute value must be less than the total number of images.
        :returns: The RGB version of the Dynamic World image.
        """
        # Check that the index is valid
        if abs(index) >= self.get_num_images():
            logging.error(
                f"Invalid index, there are only {self.get_num_images()} images in this collection."
            )
        # If the index is negative, change it to the appropriate positive value
        if index < 0:
            index += self.get_num_images()

        # Get the appropriate Dynamic World and Sentinel 2 images
        dw_s2 = self._get_dw_s2_collection()
        dw_s2_list = dw_s2.toList(dw_s2.size())
        dw_image = ee.Image(dw_s2_list.get(index))
        s2_image = ee.Image(dw_image.get("s2_img"))

        # Point geometry doesn't need to be clipped by polygon geometry does
        if type(self.geometry) is ee.Geometry.Polygon:
            dw_image = dw_image.clip(self.geometry)
            s2_image = s2_image.clip(self.geometry)

        # Add both images to the map as separate layers
        date_str = utils.get_image_date(s2_image)
        s2_label = self.IMAGE_LABEL.format(index, date_str, "Sentinel 2")
        self.add_layer(s2_image, s2_label, self.S2_VIS_PARAMS, is_dw=False)
        dw_label = self.IMAGE_LABEL.format(index, date_str, "Land cover")
        self.add_layer(dw_image, dw_label)
        # Center on the image
        self.map.center_object(dw_image.geometry())

    def map_multiple_images(self, num_layers: int = 5) -> None:
        """
        Divide the collection into different time periods and map the composite Dynamic World land 
        cover for each period as a different layer.
        :param num_layers: The number of time periods to segment into.
        :returns: A list of the RGB version of the Dynamic World composite for each period.
        """
        collecs = self.bucket_by_time(num_buckets=num_layers)
        for collec in collecs:
            comp_img = self.get_composite_image(collec)
            label = self.COMPOSITE_LABEL.format(
                utils.get_image_date(collec.first()),
                utils.get_image_date(utils.get_latest_image(collec)),
            )
            self.add_layer(comp_img, label)

    def map_composite_image(self) -> None:
        """
        Map the entire collection as a single Dynamic World composite.
        :returns: The RGB version of the Dynamic World composite.
        """
        comp_img = self.get_composite_image()
        label = self.COMPOSITE_LABEL.format(
            utils.format_date(self.start_date), utils.format_date(self.end_date)
        )
        self.add_layer(comp_img, label)

    def map_single_band(
            self, band: str, imgs: ee.ImageCollection = None, threshold: int = None
        ) -> None:
        """
        Get the composite of a collection and map one type of land cover for that composite. The
        map is masked to show where the mean probability is over some threshold, and the elevation 
        corresponds to the mean probability as well.
        :param band: Which band to map. Should be one of the bands in BANDS.
        :param imgs: The collection to take the composite of, the instance's full collection by default.
        :param threshold: The probability below which to mask, 90% of the maximum value by default.
        :returns: The RGB version of the Dynamic World composite.
        """
        if band not in self.BANDS:
            logging.error(
                f"{band} is not a valid band. Must be one of {', '.join(self.BANDS)}."
            )
        if imgs is None:
            imgs = self.collection

        band_prob = self.get_composite_image(imgs).select(band)
        if threshold is None:
            # Get the maximum probability that appears for that band across all pixels 
            # in all images
            max_val = (
                imgs.select(band)
                .reduce(ee.Reducer.max())
                .reduceRegion(reducer=ee.Reducer.max(), geometry=self.geometry)
                .values()
                .get(0)
                .getInfo()
            )
            threshold = max_val * 0.9
        # Calculate the elevation visualization from the single band's probability
        top_prob_hillshade = ee.Terrain.hillshade(
            band_prob.multiply(100 / max_val)
        ).divide(255)
        # Combine the color and elevation, masking below the threshold
        viz_img = (
            band_prob.visualize(palette=self.BAND_TO_COLOR[band])
            .multiply(top_prob_hillshade)
            .updateMask(band_prob.gt(threshold))
        )

        # Label with the land cover type and the threshold probability
        label = self.SINGLE_BAND_LABEL.format(band, threshold)
        self.add_layer(viz_img, label, is_dw=False)

    def summarize_image_by_band(
        self, img: ee.Image, as_pct: bool = True, max_px: int = 1e10
    ) -> pd.Series:
        """
        Count the pixels of each land cover type in a given image.
        :param img: The image to summarize.
        :param as_pct: Whether to normalize by the total pixel count.
        :param max_px: The maximum resolution. Cannot be more than the 1e10 default
            and a lower limit will run faster.
        :returns: A Series where the index is the land cover type and the value
            is the pixel count or percentage.
        """
        pixel_counts = (
            img.select("label")
            # Get a histogram of the label across the region
            .reduceRegion(
                reducer=ee.Reducer.frequencyHistogram().unweighted(),
                geometry=self.geometry,
                maxPixels=max_px,
                bestEffort=True, # Reduce the resolution as needed to stay below maxPixels
                scale=10,        # Dynamic World is at a 10m scale
            )
            .get("label")
            .getInfo()
        )
        if as_pct:
            total_pixels = sum(pixel_counts.values())
            pixel_counts = {
                band: count / total_pixels * 100 for band, count in pixel_counts.items()
            }
        return pd.Series(pixel_counts)

    def graph_change_in_bands(
        self, num_buckets: int = 5, max_px: int = 1e10
    ) -> alt.Chart:
        """
        Divide the collection into different time periods and get the distribution of land cover 
        types for each period to create a stacked bar chart.
        :param num_buckets: The number of time periods to segment into.
        :param max_px: The maximum resolution. Cannot be more than the 1e10 default
            and a lower limit will run faster.
        :returns: An altair stacked bar chart.
        """
        collecs = self.bucket_by_time(num_buckets=num_buckets)
        counts = []
        labels = []
        for collec in collecs:
            comp_img = self.get_composite_image(collec)
            px_count = self.summarize_image_by_band(
                comp_img, as_pct=True, max_px=max_px
            )
            label = "{} to\n{}".format(
                utils.get_image_date(collec.first())[:-6], # Remove the year
                utils.get_image_date(utils.get_latest_image(collec)),
            )
            counts.append(px_count)
            labels.append(label)

        df = (
            pd.DataFrame(counts, index=labels)
            .reset_index()
            .drop(columns="null", errors="ignore")   # I think this corresponds to cloudy pixels
            # The land cover types are represented as integers, so we map to their string names
            .rename(columns={str(i): band for i, band in enumerate(self.BANDS)})
        ).melt(id_vars="index", var_name="band", value_name="pct")

        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(
                "index:N", title="Date", sort=labels, axis=alt.Axis(labelAngle=0)
            ),
            y=alt.Y("pct:Q", title="% of the region"),
            color=alt.Color(
                "band:N",
                title="Land cover type",
                sort=self.BANDS,
                scale=alt.Scale(range=self.COLORS),
            ),
        ).properties(
            width=800,
            height=400,
            title="Change in land cover from {} to {}".format(
                utils.get_image_date(self.get_earliest_image()),
                utils.get_image_date(self.get_latest_image()),
            ),
        )
        return chart
    
    def map_volatility(self, imgs: ee.ImageCollection = None, max_val: int = 0.2) -> tuple[ee.Image]:
        """
        Maps the volatility of land cover type by taking the highest standard deviation of
        the standard devation across each band. Creates two layers: a grayscale raster, and the typical
        color map masked to pixels with std. dev. over 80% of the maximum.
        :param imgs: The image collection to analyze, the instance's entire collection by default.
        :param max_val: The high standard deviation cutoff. Corresponds to white on the grayscale map
            and the mask is 80% of this value.
        :returns: A two-element tuple of the grayscale and masked images.
        """
        if imgs is None:
            imgs = self.collection

        std_dev = (
            imgs.select(self.BANDS)
            .reduce(ee.Reducer.stdDev())
            .reduce(ee.Reducer.max())
        )
        self.map.add_layer(std_dev, {"max": max_val}, "Std. dev.")

        comp_img = self.get_composite_image(imgs)
        rgb_img = self.visualize_image(comp_img)
        std_dev_mask = std_dev.gt(max_val * 0.8).selfMask()
        masked_rgb = std_dev_mask.multiply(rgb_img)
        self.map.add_layer(masked_rgb, self.DW_VIS_PARAMS, "Composite masked by std. dev.")

        return std_dev, masked_rgb

    def save_layer_locally(self, layer_name: str, dest: str) -> None:
        utils.export_image(self.get_layer_by_name(layer_name), dest, self.geometry)

    def save_all_layers(self, dest_folder: str) -> None:
        if not os.path.exists:
            os.mkdir(dest_folder)
        for i, img in enumerate(self.layers.values()):
            dest = os.path.join(dest_folder, f"layer{i}.tif")
            utils.export_image(img, dest, self.geometry)

    def __str__(self) -> str:
        return_str += f"Count: {self.num_images} images\n"
        return_str += f"Date range: {self.start_date.format('YYYY-MM-dd').getInfo()} to {self.end_date.format('YYYY-MM-dd').getInfo()}\n"
        return_str += f"Bands: {', '.join(self.get_bands())}\n"
        return_str += f"Layers: {', '.join(self.get_all_layers().keys())}\n"
        return return_str
