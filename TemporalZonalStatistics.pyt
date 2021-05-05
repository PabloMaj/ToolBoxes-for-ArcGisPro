# -*- coding: utf-8 -*-

import arcpy
import numpy as np
from skimage.filters import threshold_otsu
import numpy.ma as ma
import os

arcpy.env.overwriteOutput = True
arcpy.env.outputCoordinateSystem = arcpy.SpatialReference("WGS 1984 UTM Zone 33N")

# Define mask based on nodata_values in bands
def define_mask(bands=None, nodata_to_value=3.4E+38):
    bands_mask = []
    for band in bands:
        mask_temp = np.logical_not(band == nodata_to_value)
        bands_mask.append(mask_temp)
    mask = np.ones(shape=mask_temp.shape)
    for mask_temp in bands_mask:
        mask = np.multiply(mask_temp, mask)
    return mask

# Check if denominator has non zero value
def check_denominator(value=None, epsilon=10**-3):
    return np.where(value == 0, epsilon, value)

# Calculate vegetation indices
def calculate_VI(R=None, G=None, B=None, RE=None, NIR=None, mask=None, raster_pattern=None, objects_to_remove=None):
    R = R.astype('float')
    G = G.astype('float')
    B = B.astype('float')
    RE = RE.astype('float')
    NIR = NIR.astype('float')
    VI = dict()
    VI['ExG'] = 2 * G - R - B
    VI["TGI"] = G - 0.39 * R - 0.61 * B
    VI['CIVE'] = -(0.441 * R - 0.881 * G + 0.385 * B + 18.787)
    VI['RGRI'] = -R / check_denominator(G)
    VI['NGRDI'] = (G - R) / check_denominator((G + R))
    VI['VARI'] = (G - R) / check_denominator((G + R - B) == 0)
    VI['VDVI'] = (2 * G - R - B) / check_denominator((2 * G + R + B))
    VI['VEG'] = G / check_denominator((np.power(R, 0.667) * np.power(B, 1 - 0.667)))
    VI['MGRVI'] = (G**2 - R**2) / check_denominator((G**2 + R**2))
    VI["RGBVI"] = (G**2 - B * R) / check_denominator((G**2 + B * R))
    VI["NDVI"] = (NIR - R) / check_denominator((NIR + R))
    for VI_name in VI.keys():
        VI[VI_name] = np.multiply(VI[VI_name], mask)
        array_to_raster(array=VI[VI_name], fileout=VI_name, blocksize=2**14, raster_pattern=raster_pattern)
        objects_to_remove.append(VI_name)
    return VI, objects_to_remove

# Convert arrays to rasters
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
        out_rasterdataset=fileout)

    # Remove temporary files
    for fileitem in filelist:
        if arcpy.Exists(fileitem):
            arcpy.Delete_management(fileitem)

class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "TemporalZonalStatistics"
        self.alias = "TemporalZonalStatistics"

        # List of tool classes associated with this toolbox
        self.tools = [Tool]


class Tool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "TemporalZonalStatistics"
        self.description = "Calculate zonal statistics for temporal multidimensional raster"
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        params = []

        rasters = arcpy.Parameter(
            displayName="Input rasters",
            name="input_rasters",
            datatype="Raster Dataset",
            parameterType="Required",
            direction="Input",
            multiValue=True)
        params.append(rasters)

        zones = arcpy.Parameter(
            displayName="Zones",
            name="zones",
            datatype="Feature Class",
            parameterType="Required",
            direction="Input")
        params.append(zones)

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
        rasters = parameters[0].valueAsText.split(";")
        zones = parameters[1].valueAsText

        objects_to_remove = []

        for counter, raster in enumerate(rasters):
            raster_pattern = arcpy.Raster(raster)
            messages.addMessage(f"Raster no {counter}")
            messages.addMessage("1. Extract individual bands")
            bands = []
            for i in [1, 2, 3, 4, 5]:
                array = arcpy.RasterToNumPyArray(
                    in_raster=arcpy.ia.ExtractBand(raster, i),
                    nodata_to_value=3.4E+38)
                bands.append(array)
            R, G, B, RE, NIR = bands

            messages.addMessage("2. Determine mask")
            nodata_to_value = list(np.unique(R))[-1]
            mask = define_mask(bands=bands, nodata_to_value=nodata_to_value)

            messages.addMessage("3. Calculate Vegetation Indices")
            VI, objects_to_remove = calculate_VI(
                R=R, G=G, B=B, RE=RE, NIR=NIR, mask=mask, raster_pattern=raster_pattern,
                objects_to_remove=objects_to_remove)

            messages.addMessage("4. Determine binary mask of segmented plants")
            masked_VDVI = ma.masked_array(VI["VDVI"], np.logical_not(mask)).compressed()
            masked_ExG = ma.masked_array(VI["ExG"], np.logical_not(mask)).compressed()
            binary_mask_VDVI = VI["VDVI"] > threshold_otsu(masked_VDVI)
            binary_mask_Exg = VI["ExG"] > threshold_otsu(masked_ExG)
            binary_mask = binary_mask_Exg * binary_mask_VDVI

            messages.addMessage("5. Binary mask of segmented plants to raster")
            array_to_raster(
                array=binary_mask.astype("int"), fileout=raster + "_binary_mask",
                blocksize=2**14, raster_pattern=raster_pattern)
            objects_to_remove.append(raster + "_binary_mask")

            messages.addMessage("6. Binary raster of segmented plants to polygons")
            arcpy.conversion.RasterToPolygon(
                in_raster=raster + "_binary_mask",
                out_polygon_features=raster + "_polygon",
                simplify="NO_SIMPLIFY",
                create_multipart_features="MULTIPLE_OUTER_PART",
                raster_field="VALUE")
            polygon_filtered = arcpy.management.SelectLayerByAttribute(
                in_layer_or_view=raster + "_polygon",
                selection_type="NEW_SELECTION",
                where_clause="gridcode = 1")
            objects_to_remove.append(raster + "_polygon")
            arcpy.management.CopyFeatures(
                in_features=polygon_filtered,
                out_feature_class=raster + "_polygon_filtered")
            objects_to_remove.append(raster + "_polygon_filtered")

            messages.addMessage("7. Clipping polygons to Region of Interest")
            arcpy.Clip_analysis(
                in_features=zones,
                clip_features=raster + "_polygon_filtered",
                out_feature_class=raster + "_polygons_in_ROI")
            objects_to_remove.append(raster + "_polygons_in_ROI")

            messages.addMessage("8. Add to output table additional names")

            fc = raster + "_polygons_in_ROI"
            fields = ["Raster_Name", "Zone_Name"]

            arcpy.management.AddField(
                in_table=fc,
                field_name=fields[0],
                field_type="TEXT")

            arcpy.management.AddField(
                in_table=fc,
                field_name=fields[1],
                field_type="TEXT")

            with arcpy.da.UpdateCursor(fc, fields) as cursor:
                for row in cursor:
                    row[0] = os.path.basename(raster)
                    row[1] = os.path.basename(zones)
                    cursor.updateRow(row)

            messages.addMessage("9. Calculate LAI")

            arcpy.management.JoinField(
                in_data=raster + "_polygons_in_ROI",
                in_field="OBJECTID",
                join_table=zones,
                join_field="OBJECTID",
                fields=["Shape_Area"])

            arcpy.management.CalculateField(
                in_table=raster + "_polygons_in_ROI",
                field="LAI",
                expression="!Shape_Area! * 100 / !Shape_Area_1!",
                expression_type="PYTHON3",
                field_type="DOUBLE")

            arcpy.management.DeleteField(
                in_table=raster + "_polygons_in_ROI",
                drop_field="Shape_Area_1")

            messages.addMessage("10. Calculate statistics in zones")
            for VI_name in VI.keys():

                arcpy.sa.ZonalStatisticsAsTable(
                    in_zone_data=raster + "_polygons_in_ROI",
                    zone_field="OBJECTID",
                    in_value_raster=VI_name,
                    out_table=VI_name + "_stats",
                    ignore_nodata="DATA",
                    statistics_type="MEAN_STD")
                objects_to_remove.append(VI_name + "_stats")

                for field in ["MEAN", "STD"]:
                    arcpy.management.AlterField(
                        in_table=VI_name + "_stats",
                        field=field,
                        new_field_name=VI_name + "_" + field,
                        new_field_alias=VI_name + "_" + field)

                arcpy.management.JoinField(
                    in_data=raster + "_polygons_in_ROI",
                    in_field="OBJECTID",
                    join_table=VI_name + "_stats",
                    join_field="OBJECTID",
                    fields=[VI_name + "_" + field for field in ["MEAN", "STD"]])

            messages.addMessage("11. Create output FeatureClass")
            fc_out = os.path.basename(raster) + "_" + os.path.basename(zones)
            arcpy.management.CopyFeatures(
                in_features=zones,
                out_feature_class=fc_out,
            )

            fields = ["LAI", "Raster_Name", "Zone_Name"]
            for VI_name in VI.keys():
                for stat in ["MEAN", "STD"]:
                    fields.append(VI_name + "_" + stat)
            arcpy.management.JoinField(
                in_data=fc_out,
                in_field="OBJECTID",
                join_table=raster + "_polygons_in_ROI",
                join_field="OBJECTID",
                fields=fields)

            for object_name in objects_to_remove:
                arcpy.management.Delete(object_name)
            messages.addMessage("Temporary files were removed")

        return 1
