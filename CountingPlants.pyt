# -*- coding: utf-8 -*-

import arcpy
import numpy as np
from skimage.morphology import binary_erosion, remove_small_objects, remove_small_holes

arcpy.env.overwriteOutput = True
arcpy.env.outputCoordinateSystem = arcpy.SpatialReference("WGS 1984 UTM Zone 33N")

def array_to_raster(array=None, fileout=None, blocksize=None, raster_pattern=None):
    filelist = []
    blockno = 0
    for x in range(0, raster_pattern.width, blocksize):
        for y in range(0, raster_pattern.height, blocksize):
            mx = raster_pattern.extent.XMin + x * raster_pattern.meanCellWidth
            my = raster_pattern.extent.YMin + y * raster_pattern.meanCellHeight
            x_start = x
            y_end = raster_pattern.height - y
            x_end = min([x_start + blocksize, raster_pattern.width])
            y_start = max([y_end - blocksize, 0])
            raster_pattern_block = arcpy.NumPyArrayToRaster(
                in_array=array[y_start:y_end, x_start:x_end],
                lower_left_corner=arcpy.Point(mx, my),
                x_cell_size=raster_pattern.meanCellWidth,
                y_cell_size=raster_pattern.meanCellHeight
            )

            filetemp = fileout + f"_{blockno}"
            raster_pattern_block.save(filetemp)
            filelist.append(filetemp)
            blockno += 1

    # Mosaic temporary files
    if len(filelist) > 1:
        raster_out = arcpy.Mosaic_management(';'.join(filelist[1:]), filelist[0])
    else:
        raster_out = filelist[0]
    arcpy.management.CopyRaster(
        in_raster=raster_out,
        out_rasterdataset=fileout,
        pixel_type="8_BIT_UNSIGNED")

    # Remove temporary files
    for fileitem in filelist:
        if arcpy.Exists(fileitem):
            arcpy.Delete_management(fileitem)

class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Counting_plants"
        self.alias = "counting_plants"

        # List of tool classes associated with this toolbox
        self.tools = [Tool]


class Tool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Counting plants"
        self.description = "Counting plants with classic counting of blobs (connected components)."
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        params = []

        vegetation_index = arcpy.Parameter(
            displayName="Vegetation index",
            name="Vegetation index",
            datatype="Raster Dataset",
            parameterType="Required",
            direction="Input"
        )
        params.append(vegetation_index)

        min_component_size = arcpy.Parameter(
            displayName="Min component size (pixels)",
            name="Min component size (pixels)",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        )
        params.append(min_component_size)

        make_hot_spot_analysis = arcpy.Parameter(
            displayName="Make hot spot analysis",
            name="Make hot spot analysis",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input"
        )
        params.append(make_hot_spot_analysis)

        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""

        # input parameters
        vegetation_index = parameters[0].valueAsText
        min_component_size = parameters[1].valueAsText
        make_hot_spot_analysis = bool(parameters[2].valueAsText)
        try:
            min_component_size = int(min_component_size)
        except:
            min_component_size = 0

        objects_to_remove = []

        # Otsu thresholding
        binary_raster = arcpy.ia.Threshold(vegetation_index)
        messages.addMessage("1. Otsu thresholding")

        # Raster to NumpyArray
        array_as_raster = arcpy.RasterToNumPyArray(
            in_raster=binary_raster,
            nodata_to_value=0)
        messages.addMessage("2. Binary mask to NumpyArray")

        # Filtering binary array
        array_as_raster = remove_small_objects(
            array_as_raster.astype("bool"), min_size=min_component_size, connectivity=1)
        array_as_raster = remove_small_holes(
            array_as_raster.astype("bool"), area_threshold=min_component_size, connectivity=1)
        messages.addMessage("3. Filtering binary mask")

        # NumpyArray to raster after filtering
        raster_pattern = arcpy.Raster(vegetation_index)
        array_to_raster(
            array=array_as_raster.astype("int"), fileout="Raster_after_process",
            blocksize=2**14, raster_pattern=raster_pattern)
        objects_to_remove.append("Raster_after_process")
        messages.addMessage("4. Rasterize filtered NumpyArray")

        # Convert filtered raster to polygons
        arcpy.conversion.RasterToPolygon(
            in_raster="raster_after_process",
            out_polygon_features="plant_polygons",
            simplify="NO_SIMPLIFY",
            raster_field="Value",
            create_multipart_features="SINGLE_OUTER_PART")
        objects_to_remove.append("plant_polygons")
        messages.addMessage("5. Convert rasterized NumpyArray to polygons")

        # Points from polygons
        arcpy.management.FeatureToPoint(
            in_features="plant_polygons",
            out_feature_class="plant_points",
            point_location="INSIDE")
        messages.addMessage("6. Create points from polygons")

        # Counting plants
        no_of_plants = arcpy.management.GetCount("plant_points")
        messages.addMessage(f"7. Number of plants in region of interest:{no_of_plants}")

        # Hot spot analysis
        if make_hot_spot_analysis:
            arcpy.stats.OptimizedHotSpotAnalysis(
                Input_Features="plant_points",
                Output_Features="plants_hot_spot_analysis",
                Incident_Data_Aggregation_Method="COUNT_INCIDENTS_WITHIN_FISHNET_POLYGONS")
            messages.addMessage("8. Hot spot analysis")

        # Remove temporary files
        for object_name in objects_to_remove:
            arcpy.management.Delete(object_name)
        messages.addMessage("Temporary files were removed")

        return 0
