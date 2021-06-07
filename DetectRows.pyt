# -*- coding: utf-8 -*-

import arcpy
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass, distance_transform_edt
from scipy.signal import find_peaks
import numpy as np
import cv2
from skimage.morphology import remove_small_holes, remove_small_objects, binary_erosion
from skimage.measure import label, regionprops
from sklearn.linear_model import LinearRegression
from scipy import ndimage
import random
from sklearn.covariance import LedoitWolf
from scipy.stats import linregress
import pandas as pd
import os
from itertools import compress

arcpy.env.overwriteOutput = True

def projekt_point_to_line(x0=None, y0=None, a=None, b=None):
    x = (y0 + x0/a - b)/(a + 1/a)
    y = a*x + b
    return x, y

def convert_array_cords_to_geo_cords(x=None, y=None, raster_pattern=None):
    y_min_raster = raster_pattern.extent.XMin
    x_min_raster = raster_pattern.extent.YMin
    cell_size = np.mean([raster_pattern.meanCellWidth, raster_pattern.meanCellHeight])
    height = raster_pattern.height
    x_out = x_min_raster + (height - x) * cell_size
    y_out = y_min_raster + y * cell_size
    return (x_out, y_out)

class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "DetectRows"
        self.alias = "detectRows"

        # List of tool classes associated with this toolbox
        self.tools = [Tool]


class Tool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "DetectRows"
        self.description = "Detect rows in binary image (after thresholding)"
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        params = []

        binary_mask = arcpy.Parameter(
            displayName="Binary mask",
            name="Binary mask",
            datatype="Raster Dataset",
            parameterType="Required",
            direction="Input"
        )
        params.append(binary_mask)

        mean_distance_between_rows = arcpy.Parameter(
            displayName="Mean distance between rows (pixels)",
            name="Mean distance between rows (pixels)",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        params.append(mean_distance_between_rows)

        mean_distance_between_plants_in_row = arcpy.Parameter(
            displayName="Mean distance between plants in row (pixels)",
            name="Mean distance between plants in row (pixels)",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        params.append(mean_distance_between_plants_in_row)

        min_number_of_plants_in_row = arcpy.Parameter(
            displayName="Min number of plants in row",
            name="Min number of plants in row",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        params.append(min_number_of_plants_in_row)

        angle = arcpy.Parameter(
            displayName="Orientation of row (degrees)",
            name="Orientation of row (degrees)",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        params.append(angle)

        object_size_to_remove = arcpy.Parameter(
            displayName="Min object size (pixels)",
            name="Min object size (pixels)",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        )
        params.append(object_size_to_remove)

        holes_size_to_remove = arcpy.Parameter(
            displayName="Min hole size (pixels)",
            name="Min hole size (pixels)",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        )
        params.append(holes_size_to_remove)

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

        # input parameters
        binary_mask_name = parameters[0].valueAsText
        mean_distance_between_rows = float(parameters[1].valueAsText.replace(",", "."))
        mean_distance_between_plants_in_row = float(parameters[2].valueAsText.replace(",", "."))
        min_number_of_plants_in_row = int(parameters[3].valueAsText)
        angle = float(parameters[4].valueAsText.replace(",", "."))
        object_size_to_remove = parameters[5].valueAsText
        holes_size_to_remove = parameters[6].valueAsText
        try:
            object_size_to_remove = int(object_size_to_remove)
        except:
            object_size_to_remove = 0
        try:
            holes_size_to_remove = int(holes_size_to_remove)
        except:
            holes_size_to_remove = 0

        # 0) Read Parameters
        messages.addMessage("0. Read parameters")
        messages.addMessage(f"mean_distance_between_rows:{mean_distance_between_rows}")
        messages.addMessage(f"mean_distance_between_plants_in_row:{mean_distance_between_plants_in_row}")
        messages.addMessage(f"min_number_of_plants_in_row:{min_number_of_plants_in_row}")
        messages.addMessage(f"angle:{angle}")
        messages.addMessage(f"object_size_to_remove:{object_size_to_remove}")
        messages.addMessage(f"holes_size_to_remove:{holes_size_to_remove}")

        # 1) Preparing binary mask to extract points
        messages.addMessage("1. Preparing binary mask to extract points")
        binary_mask = arcpy.RasterToNumPyArray(in_raster=binary_mask_name, nodata_to_value=0).astype("bool")
        binary_mask = remove_small_objects(binary_mask, object_size_to_remove)
        binary_mask = remove_small_holes(binary_mask, holes_size_to_remove)
        distance = distance_transform_edt(binary_mask)
        binary_mask = (distance > np.mean([i for i in distance.flatten() if i != 0]))

        """
        plt.clf()
        plt.imshow(binary_mask)
        plt.savefig("processing_steps\\binary_mask_after_cleaning.png", dpi=300)
        """

        # 2) Extract points
        messages.addMessage("2. Extract center points of blobs and calculating properties of blobs")
        clusters, n_clusters = label(binary_mask, background=0, return_num=True)
        props = regionprops(clusters)
        clusters_center = [prop['centroid'] for prop in props]
        clusters_major_axis = [prop['major_axis_length'] for prop in props]
        clusters_cords = [prop['coords'] for prop in props]
        cluster_ids = np.arange(n_clusters)
        points_all = clusters_center

        # 3) Calculate orientation of blobs
        messages.addMessage("3. Calculate orientation of blobs")
        clusters_orientation_eigen = []
        for i in range(len(clusters_cords)):
            clusters_cords_normalized = clusters_cords[i] - np.mean(clusters_cords[i], axis=0).astype("int")
            if clusters_cords_normalized.shape[0] < 2:
                theta = angle
                clusters_orientation_eigen.append(theta)
                continue
            cov = LedoitWolf().fit(clusters_cords_normalized).covariance_
            print(clusters_cords_normalized.shape)
            evals, evecs = np.linalg.eig(cov)
            sort_indices = np.argsort(evals)[::-1]
            x_v1, y_v1 = evecs[:, sort_indices[0]]
            x_v2, y_v2 = evecs[:, sort_indices[1]]
            if x_v1 == 0:
                x_v1 = 10**(-5)
            theta = np.arctan((y_v1) / (x_v1))
            clusters_orientation_eigen.append(theta)

        # 4) Add extra points
        messages.addMessage("4. Add extra points")
        rows_candidates_ids = cluster_ids[np.array(clusters_major_axis) > 2 * mean_distance_between_plants_in_row]
        for cluster_id in rows_candidates_ids:
            a = np.tan(clusters_orientation_eigen[cluster_id])
            b = clusters_center[cluster_id][1] - a * clusters_center[cluster_id][0]

            step_x = int(np.cos(clusters_orientation_eigen[cluster_id]) * mean_distance_between_plants_in_row)
            step_y = int(np.sin(clusters_orientation_eigen[cluster_id]) * mean_distance_between_plants_in_row)
            no_new_points_new_side = int(clusters_major_axis[cluster_id] / (2 * mean_distance_between_plants_in_row)) - 1

            new_points = []
            for i in range(1, no_new_points_new_side + 1):
                x_1 = int(clusters_center[cluster_id][0] + i * step_x)
                y_1 = int(clusters_center[cluster_id][1] + i * step_y)
                x_2 = int(clusters_center[cluster_id][0] - i * step_x)
                y_2 = int(clusters_center[cluster_id][1] - i * step_y)
                new_points.append((x_1, y_1))
                new_points.append((x_2, y_2))

            points_all += new_points

        """
        image = np.dstack((binary_mask * 255, binary_mask * 255, binary_mask * 255))
        for point in clusters_center:
            point_ = (int(point[1]), int(point[0]))
            image = cv2.circle(image, point_, radius=5, color=(0, 255, 0), thickness=-1)
        plt.clf()
        plt.imshow(image)
        plt.savefig("processing_steps\\binary_mask_with_all_important_points_of_blobs.png", dpi=300)
        """

        # 5) Find lines
        messages.addMessage("5. Find line")
        clusters_center_copy = set(list(clusters_center))
        # clusters_cords_copy = clusters_cords
        # map_of_plants = np.zeros(shape=binary_mask.shape)
        # counter_plant_in_row = 1
        border_points = []
        while len(clusters_center_copy) != 0:

            point_example = random.choice(list(clusters_center_copy))
            a = np.tan(angle * np.pi / 180)
            b = point_example[1] - a * point_example[0]
            A = -a
            B = 1
            C = -b
            coefs = np.array([A, B, C])

            distances = np.multiply(np.array(np.array(list(clusters_center_copy))), coefs[:2])
            distances = np.sum(distances, axis=1)
            distances += C
            distances = np.abs(distances)
            distances /= np.sqrt(A**2 + B**2)

            mask = distances <= mean_distance_between_rows / 4

            points_choosen = np.array(np.array(list(clusters_center_copy)))[mask, :]
            x = list(points_choosen[:, 0])
            y = list(points_choosen[:, 1])
            if len(x) > 1:
                a_line, b_line, r, p, se = linregress(x, y)
            else:
                r = 0

            """
            messages.addMessage(mask)
            cords_choosen = list(compress(data=clusters_cords_copy, selectors=mask))
            clusters_cords_copy = list(compress(data=clusters_cords_copy, selectors=np.logical_not(mask)))
            messages.addMessage(len(cords_choosen))
            messages.addMessage(len(clusters_cords_copy))
            """

            if points_choosen.shape[0] < min_number_of_plants_in_row or np.abs(r) < 0.99:
                list_of_tuples = set(tuple(i) for i in points_choosen.tolist())
                clusters_center_copy = clusters_center_copy.difference(list_of_tuples)

                """
                for cords in cords_choosen:
                    for one_plant_cords in cords:
                        x_, y_ = one_plant_cords
                        map_of_plants[x_, y_] = -1
                """
                continue

            else:
                x_center, y_center = np.mean(points_choosen, axis=0)
                dist = np.linalg.norm(points_choosen - np.array([x_center, y_center]), axis=1)
                x1, y1 = list(points_choosen)[np.argmax(dist)]
                dist = np.linalg.norm(points_choosen - np.array([x1, y1]), axis=1)
                x2, y2 = list(points_choosen)[np.argmax(dist)]

                x1, y1 = projekt_point_to_line(x0=x1, y0=y1, a=a_line, b=b_line)
                x2, y2 = projekt_point_to_line(x0=x2, y0=y2, a=a_line, b=b_line)

                border_points.append([(x1, y1), (x2, y2)])
                list_of_tuples = set(tuple(i) for i in points_choosen.tolist())
                clusters_center_copy = clusters_center_copy.difference(list_of_tuples)

                """
                for cords in cords_choosen:
                    for one_plant_cords in cords:
                        x_, y_ = one_plant_cords
                        map_of_plants[x_, y_] = counter_plant_in_row
                        counter_plant_in_row += 1
                """
        """
        image = np.dstack((binary_mask * 255, binary_mask * 255, binary_mask * 255))
        for points in border_points:
            point_1, point_2 = points
            point_1 = (int(point_1[1]), int(point_1[0]))
            point_2 = (int(point_2[1]), int(point_2[0]))
            image = cv2.circle(image, point_1, radius=10, color=(0, 255, 0), thickness=-1)
            image = cv2.circle(image, point_2, radius=10, color=(0, 255, 0), thickness=-1)
            cv2.line(image, point_1, point_2, (255, 0, 0), thickness=2)
        plt.clf()
        plt.imshow(image)
        plt.savefig("processing_steps\\binary_mask_with_lines_and_border_points.png", dpi=300)
        """

        """
        plt.clf()
        plt.imshow(map_of_plants)
        plt.colorbar()
        plt.savefig("processing_steps\\map_of_plants.png", dpi=300)
        """

        # 6) Find border points and save to *.csv
        messages.addMessage("6. Find border points and save to *.csv")
        raster_pattern = arcpy.Raster(binary_mask_name)
        rows = []
        for counter, pt_pair in enumerate(border_points):
            pt1, pt2 = pt_pair
            pt1_converted = convert_array_cords_to_geo_cords(x=pt1[0], y=pt1[1], raster_pattern=raster_pattern)
            pt2_converted = convert_array_cords_to_geo_cords(x=pt2[0], y=pt2[1], raster_pattern=raster_pattern)

            row = dict()
            row["coord_N"] = pt1_converted[0]
            row["coord_E"] = pt1_converted[1]
            row["line_id"] = counter + 1
            rows.append(row)

            row = dict()
            row["coord_N"] = pt2_converted[0]
            row["coord_E"] = pt2_converted[1]
            row["line_id"] = counter + 1
            rows.append(row)

        df = pd.DataFrame(rows)
        try:
            os.remove("border_points.csv")
        except:
            pass
        df.to_csv("border_points.csv", index=False)

        # 7) Convert coords of points from csv to Point FeatureClass
        try:
            arcpy.Delete("border_points")
        except:
            pass
        messages.addMessage("7. Convert coords of points from csv to Point FeatureClass")
        arcpy.management.XYTableToPoint(
            in_table=os.path.dirname(os.path.realpath("border_points.csv")) + f"\\border_points.csv",
            out_feature_class="border_points",
            x_field="coord_E",
            y_field="coord_N",
            z_field=None,
            coordinate_system=arcpy.Describe(raster_pattern).spatialReference)

        # 8) Convert Point FeatureClass to Line
        try:
            arcpy.Delete("lines")
        except:
            pass
        messages.addMessage("8. Convert Point FeatureClass to Line")
        arcpy.management.PointsToLine(
            Input_Features="border_points",
            Output_Feature_Class="lines",
            Line_Field="line_id",
            Close_Line="NO_CLOSE")

        return
