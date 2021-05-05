# -*- coding: utf-8 -*-

import arcpy
import numpy as np
import random
from numba import jit
from skimage.transform import resize
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

arcpy.env.overwriteOutput = True
arcpy.env.outputCoordinateSystem = arcpy.SpatialReference("WGS 1984 UTM Zone 33N")

# Define mask based on nodata_values in bands
def define_mask(bands=None, bands_names=None, nodata_to_value=None):
    bands_mask = []
    for band_name in bands_names:
        mask_temp = np.logical_not(bands[band_name] == nodata_to_value)
        bands_mask.append(mask_temp)
    mask = np.ones(shape=mask_temp.shape)
    for mask_temp in bands_mask:
        mask = np.multiply(mask_temp, mask)
    return mask

# Choose pixels for analysis with binary mask
def choose_pixels_with_mask(bands=None, ground_truth=None, mask=None, is_RGB_map=None):
    X = []
    Y = []
    x_values, y_values = np.where(mask == 1)
    no_values = x_values.shape[0]
    for i in range(no_values):
        x = x_values[i]
        y = y_values[i]
        if is_RGB_map:
            X.append(bands["RGB"][x, y])
        else:
            X.append(bands[x, y])
        Y.append(ground_truth[x, y])
    X = np.array(X)
    Y = np.array(Y)

    return X, Y

# Calculate optimal threshold (working point) for optimised indicator
def calculate_optimal_threshold(VI=None, Y_true=None, repeats=5, n=20, threshold_start=0, threshold_end=1):
    for _ in range(repeats):
        results = []
        for threshold in np.linspace(start=threshold_start, stop=threshold_end, num=n):
            results.append((threshold, f1_score((VI > threshold), Y_true)))
        item_max = max(results, key=lambda x: x[1])
        threshold_opt = item_max[0]
        F1_max = item_max[1]
        pos_max = results.index(item_max)
        pos_max = min(n - 2, pos_max)
        pos_max = max(1, pos_max)
        threshold_start = results[pos_max - 1][0]
        threshold_end = results[pos_max + 1][0]
    return threshold_opt, F1_max

# PSO (Particle Swarm Optimization)
class Particle():
    def __init__(self):
        self.position = np.random.uniform(-2, 2, 6)
        self.pbest_position = self.position
        self.pbest_value = float('inf')
        self.velocity = np.random.uniform(0, 0, 6)

    def move(self):
        self.position = self.position + self.velocity
class Space():

    def __init__(self, n_particles, X_train, Y_train):
        self.n_particles = n_particles
        self.particles = []
        self.gbest_value = float('inf')
        self.gbest_position = []
        self.X_train = X_train
        self.Y_train = Y_train

    def fitness(self, particle):
        w_up = particle.position[:3]
        w_down = particle.position[3:]
        VI = calculate_VI_fraction(X=self.X_train, w_up=w_up, w_down=w_down)
        # F1_score = calculate_F1(VI=VI, threshold=0.2, ground_truth=self.Y_train)
        AUC = roc_auc_score(y_true=self.Y_train, y_score=VI)
        return 1 - AUC

    def set_pbest(self):
        for particle in self.particles:
            fitness_cadidate = self.fitness(particle)
            if(particle.pbest_value > fitness_cadidate):
                particle.pbest_value = fitness_cadidate
                particle.pbest_position = particle.position
            if(self.gbest_value > fitness_cadidate):
                self.gbest_value = fitness_cadidate
                self.gbest_position = particle.position
                """
                print("---------------------------------------")
                print(f"Best position:{self.gbest_position}\n")
                print(f"Max F1_score:{1-self.gbest_value}\n")
                print("---------------------------------------")
                """

    def move_particles(self):
        W = 0.9
        c1 = 2
        c2 = 2
        for particle in self.particles:
            new_velocity = (W * particle.velocity) + (c1 * random.random()) *\
                (particle.pbest_position - particle.position) + (random.random() * c2) *\
                (self.gbest_position - particle.position)
            particle.velocity = new_velocity
            particle.move()

# Calculate fraction indicator based on given parameters
@jit(nopython=True, parallel=True)
def calculate_VI_fraction(X=None, w_up=None, w_down=None):
    epsilon = 10**-6
    VI = np.sum(X * w_up, axis=1) / (np.sum(X * w_down, axis=1) + epsilon)
    return VI

# Calculate F1-score between thresholded indicator and ground truth
@jit(nopython=True, parallel=True)
def calculate_F1(VI=None, threshold=0.5, ground_truth=None):
    mask_predicted = (VI > threshold)
    TP = np.multiply((mask_predicted == ground_truth), ground_truth == 1)
    FP = np.multiply((mask_predicted != ground_truth), ground_truth == 1)
    FN = np.multiply((mask_predicted != ground_truth), ground_truth == 0)
    TP = np.sum(TP)
    FP = np.sum(FP)
    FN = np.sum(FN)
    F1_score = 2 * TP / (2 * TP + FP + FN)
    return F1_score

# Check coordinate system of Rasters and FeatureClass
def check_coord_system(objects_to_remove=None, parameters=None, messages=None):
    objects_to_remove = []
    # Zmiana układu współrzędnych na jednolity
    for counter, param in enumerate(parameters):
        if type(param.valueAsText) == str:
            if str(arcpy.Describe(param.valueAsText).spatialReference.name) != "WGS_1984_UTM_Zone_33N":
                # messages.addMessage(str(arcpy.Describe(param.valueAsText).spatialReference.name))
                if param.datatype == "Feature Class":
                    arcpy.management.Project(
                        in_dataset=param.valueAsText,
                        out_dataset=param.valueAsText + "_projected",
                        out_coor_system=arcpy.SpatialReference("WGS 1984 UTM Zone 33N"))
                elif param.datatype == "Raster Dataset":
                    arcpy.management.ProjectRaster(
                        in_raster=param.valueAsText,
                        out_raster=param.valueAsText + "_projected",
                        out_coor_system=arcpy.SpatialReference("WGS 1984 UTM Zone 33N"))
                parameters[counter].value = param.valueAsText + "_projected"
                objects_to_remove.append(parameters[counter].valueAsText)

    return parameters, messages, objects_to_remove

# Read data (extract bands and convert to Numpy arrays)
def read_data(raster_in=None, messages=None):
    bands = dict()
    arrays = dict()
    bands["Red"] = arcpy.ia.ExtractBand(raster=raster_in, band_ids=1)
    bands["Green"] = arcpy.ia.ExtractBand(raster=raster_in, band_ids=2)
    bands["Blue"] = arcpy.ia.ExtractBand(raster=raster_in, band_ids=3)
    arrays["Red"] = arcpy.RasterToNumPyArray(in_raster=bands["Red"]).astype("float")
    arrays["Green"] = arcpy.RasterToNumPyArray(in_raster=bands["Green"]).astype("float")
    arrays["Blue"] = arcpy.RasterToNumPyArray(in_raster=bands["Blue"]).astype("float")
    # messages.addMessage(str(arrays["Red"].shape))
    return messages, bands, arrays

# Clip raster to Region of interest (ROI)
def clip_raster_to_ROI(bands=None, bands_names=None, ROI=None, messages=None, label=None, objects_to_remove=None):
    bands_out = dict()
    # finding extent
    for row in arcpy.da.SearchCursor(ROI, ['SHAPE@']):
        extent = row[0].extent
        break
    for band_name in bands_names:
        arcpy.management.Clip(
            in_raster=bands[band_name],
            out_raster=f'{band_name}_clipping_{label}',
            in_template_dataset=ROI,
            rectangle=f"{extent.XMin} {extent.YMin} {extent.XMax} {extent.YMax}",
            nodata_value=3.4E+38,
            clipping_geometry='ClippingGeometry',
            maintain_clipping_extent='MAINTAIN_EXTENT')
        bands_out[band_name] = arcpy.Raster(f'{band_name}_clipping_{label}')
        objects_to_remove.append(f'{band_name}_clipping_{label}')
    # arcpy.management.CopyRaster(bands_out[band_name], f"Crop_{label}_example")
    return bands_out, objects_to_remove

# Rasterize GT in the form of FeatureClass to Raster format
def rasterize_GT(labels_segmentation=None, ROI=None, cell_size=None, messages=None, label=None, objects_to_remove=None):
    arcpy.conversion.FeatureToRaster(
        in_features=labels_segmentation,
        field="OBJECTID",
        out_raster=f"GT_raster_{label}",
        cell_size=cell_size)
    objects_to_remove.append(f"GT_raster_{label}")

    # finding extent
    for row in arcpy.da.SearchCursor(ROI, ['SHAPE@']):
        extent = row[0].extent
        break

    arcpy.management.Clip(
        in_raster=f"GT_raster_{label}",
        out_raster=f"GT_raster_{label}_Clip",
        in_template_dataset=ROI,
        rectangle=f"{extent.XMin} {extent.YMin} {extent.XMax} {extent.YMax}",
        nodata_value=0,
        clipping_geometry='ClippingGeometry',
        maintain_clipping_extent='MAINTAIN_EXTENT')
    objects_to_remove.append(f"GT_raster_{label}_Clip")

    return objects_to_remove, messages

# Convert rasters (GT and bands) to numpy arrays
def convert_rasters_to_numpy(bands=None, bands_names=None, label=None, messages=None):
    arrays = dict()
    lower_left_corner = arcpy.Point(bands["Red"].extent.XMin, bands["Red"].extent.YMin)
    for band_name in bands_names:
        arrays[band_name] = arcpy.RasterToNumPyArray(
            in_raster=bands[band_name],
            lower_left_corner=lower_left_corner,
            nodata_to_value=3.4E+38)
    arrays["RGB"] = np.dstack((arrays["Red"], arrays["Green"], arrays["Blue"]))
    arrays["GT"] = arcpy.RasterToNumPyArray(
        in_raster=f"GT_raster_{label}_Clip",
        lower_left_corner=lower_left_corner,
        nodata_to_value=0)
    return arrays

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
        out_rasterdataset=fileout,
        pixel_type="32_BIT_FLOAT")

    # Remove temporary files
    for fileitem in filelist:
        if arcpy.Exists(fileitem):
            arcpy.Delete_management(fileitem)

# resize GT to size of bands (removing small difference after preprocessing)
def resize_GT(arrays=None):
    arrays["GT"] = arrays["GT"].astype("float")
    arrays["GT"] = resize(arrays["GT"], (arrays["Red"].shape[0], arrays["Red"].shape[1]))
    arrays["GT"] = (arrays["GT"] > 0.5)
    return arrays

class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Toolbox"
        self.alias = "toolbox"

        # List of tool classes associated with this toolbox
        self.tools = [Tool]

class Tool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Optimised Indicators"
        self.description = ""
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        params = []

        raster_in = arcpy.Parameter(
            displayName="Input Raster",
            name="raster_in",
            datatype="Raster Dataset",
            parameterType="Required",
            direction="Input"
        )
        params.append(raster_in)

        labels_segmentation_train = arcpy.Parameter(
            displayName="Labels for train for segmentation",
            name="labels_segmentation_train",
            datatype="Feature Class",
            parameterType="Required",
            direction="Input"
        )
        params.append(labels_segmentation_train)

        ROI_segmentation_train = arcpy.Parameter(
            displayName="ROI for train for segmentation",
            name="ROI_segmentation_train",
            datatype="Feature Class",
            parameterType="Required",
            direction="Input"
        )
        params.append(ROI_segmentation_train)

        labels_segmentation_test = arcpy.Parameter(
            displayName="Labels for test for segmentation",
            name="labels_segmentation_test",
            datatype="Feature Class",
            parameterType="Optional",
            direction="Input"
        )
        params.append(labels_segmentation_test)

        ROI_segmentation_test = arcpy.Parameter(
            displayName="ROI for test for segmentation",
            name="ROI_segmentation_test",
            datatype="Feature Class",
            parameterType="Optional",
            direction="Input"
        )
        params.append(ROI_segmentation_test)

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

        # check coordinate system (WGS 1984 UTM Zone 33N)
        parameters, messages, objects_to_remove = check_coord_system(
            objects_to_remove=[], parameters=parameters, messages=messages)

        # input parameters
        raster_in = parameters[0].valueAsText
        labels_segmentation_train = parameters[1].valueAsText
        ROI_segmentation_train = parameters[2].valueAsText
        labels_segmentation_test = parameters[3].valueAsText
        ROI_segmentation_test = parameters[4].valueAsText

        # read data
        messages, bands, arrays = read_data(raster_in=raster_in, messages=messages)

        messages.addMessage("1. Data was readed")

        bands_names = ["Red", "Green", "Blue"]
        # clip raster to train ROI
        train_bands, objects_to_remove = clip_raster_to_ROI(
            bands=bands, bands_names=bands_names, ROI=ROI_segmentation_train, messages=messages,
            label="train", objects_to_remove=objects_to_remove)

        # clip raster to test ROI (optional)
        if isinstance(ROI_segmentation_test, str):
            test_bands, objects_to_remove = clip_raster_to_ROI(
                bands=bands, bands_names=bands_names, ROI=ROI_segmentation_test, messages=messages,
                label="test", objects_to_remove=objects_to_remove)

        messages.addMessage("2. Rasters was clipped to ROI")

        # Rasterize GT for train
        objects_to_remove, messages = rasterize_GT(
            labels_segmentation=labels_segmentation_train, ROI=ROI_segmentation_train, cell_size=bands["Red"],
            messages=messages, label="train", objects_to_remove=objects_to_remove)

        # Rasterize GT for test (optional)
        if isinstance(labels_segmentation_test, str) and isinstance(ROI_segmentation_test, str):
            objects_to_remove, messages = rasterize_GT(
                labels_segmentation=labels_segmentation_test, ROI=ROI_segmentation_test, cell_size=bands["Red"],
                messages=messages, label="test", objects_to_remove=objects_to_remove)

        messages.addMessage("3. Ground Truth was rasterized")

        # Train bands and GT to numpy arrays
        train_arrays = convert_rasters_to_numpy(
            bands=train_bands, bands_names=bands_names, label="train", messages=messages)

        # Test bands and GT to numpy arrays
        if isinstance(labels_segmentation_test, str):
            test_arrays = convert_rasters_to_numpy(
                bands=test_bands, bands_names=bands_names, label="test", messages=messages)

        messages.addMessage("Band_shape:" + str(train_arrays["Red"].shape) +
                            "\nGT_shape:" + str(train_arrays["GT"].shape))

        messages.addMessage("4. Ground Truth and bands were converted to numpy arrays")

        # Define binary mask for train
        nodata_to_value = list(np.unique(train_arrays["Red"][0][-1]))[0]
        train_mask = define_mask(bands=train_arrays, bands_names=bands_names, nodata_to_value=nodata_to_value)

        # Define binary mask for test
        if isinstance(labels_segmentation_test, str):
            test_mask = define_mask(bands=test_arrays, bands_names=bands_names, nodata_to_value=nodata_to_value)

        messages.addMessage("5. Mask was created")

        # Resize GT to size of bands
        train_arrays = resize_GT(arrays=train_arrays)

        if isinstance(labels_segmentation_test, str):
            test_arrays = resize_GT(arrays=test_arrays)

        messages.addMessage("6. Ground Truth was resized")

        # Choose individual pixels for training models
        X_train, Y_train = choose_pixels_with_mask(
            bands=train_arrays, ground_truth=train_arrays["GT"], mask=train_mask, is_RGB_map=1)

        if isinstance(labels_segmentation_test, str):
            X_test, Y_test = choose_pixels_with_mask(
                bands=test_arrays, ground_truth=test_arrays["GT"], mask=test_mask, is_RGB_map=1)

        messages.addMessage("7. Pixels were choosen with binary mask")

        # Optimised linear indicator
        clf = LinearRegression().fit(X_train, Y_train)

        messages.addMessage("8. Optimatisation of linear indicator was finished")
        messages.addMessage(f"VI = {np.round(clf.coef_[0], 4)}*R + {np.round(clf.coef_[1], 4)}*G +"
                            f"{np.round(clf.coef_[2], 4)}*B + {np.round(clf.intercept_, 4)}")

        # Evaluation of optimised linear indicator
        messages.addMessage("9. Optimised linear indicators evaluation:")

        Y_predict_train = clf.predict(X_train)
        threshold_opt, _ = calculate_optimal_threshold(
            VI=Y_predict_train, Y_true=Y_train, repeats=5, n=20, threshold_start=0, threshold_end=1)
        Y_predict_train = (Y_predict_train > threshold_opt)
        F1_score_train = f1_score(Y_predict_train, Y_train)
        F1_score_train = np.round(F1_score_train, 4)
        messages.addMessage(f"F1_score_train:{F1_score_train}")

        if isinstance(labels_segmentation_test, str):
            Y_predict_test = clf.predict(X_test)
            Y_predict_test = (Y_predict_test > threshold_opt)
            F1_score_test = f1_score(Y_predict_test, Y_test)
            F1_score_test = np.round(F1_score_test, 4)
            messages.addMessage(f"F1_score_test:{F1_score_test}")

        # Export raster with novel linear indicator
        array_out_linear_optimised = clf.coef_[0] * arrays["Red"] + clf.coef_[1] * arrays["Green"] +\
            clf.coef_[2] * arrays["Blue"] + clf.intercept_
        array_to_raster(
            array=array_out_linear_optimised, fileout="Optimised_linear_indicator",
            blocksize=2**13, raster_pattern=bands["Red"])

        messages.addMessage("10. Raster for optimised linear indicators was exported")

        # Optimised fraction indicator
        n_iterations = 20
        n_particles = 30
        search_space = Space(n_particles, X_train, Y_train)
        particles_vector = [Particle() for _ in range(search_space.n_particles)]
        search_space.particles = particles_vector

        iteration = 0
        while(iteration < n_iterations):
            search_space.set_pbest()
            search_space.move_particles()
            iteration += 1

        coefs_best = np.array(search_space.gbest_position)
        scaling = np.sum(np.abs(coefs_best))
        coefs_best /= scaling
        w_up = coefs_best[:3]
        w_down = coefs_best[3:]

        messages.addMessage("11. Optimatisation of fraction indicator was finished")
        messages.addMessage(f"VI = ({np.round(w_up[0], 4)}*R + {np.round(w_up[1], 4)}*G + {np.round(w_up[2], 4)}*B) /"
                            f"({np.round(w_down[0], 4)}*R + {np.round(w_down[1], 4)}*G + {np.round(w_down[2], 4)}*B)")

        # Evaluation of optimised fraction indicator
        messages.addMessage("12. Optimised fraction indicators evaluation:")

        VI_train = calculate_VI_fraction(X=X_train, w_down=w_down, w_up=w_up)
        threshold_opt, _ = calculate_optimal_threshold(
            VI=VI_train, Y_true=Y_train, repeats=5, n=20, threshold_start=-5, threshold_end=5)
        Y_predict_train = (VI_train > threshold_opt)
        F1_score_train = f1_score(Y_predict_train, Y_train)
        F1_score_train = np.round(F1_score_train, 4)
        messages.addMessage(f"F1_score_train:{F1_score_train}")

        if isinstance(labels_segmentation_test, str):
            VI_test = calculate_VI_fraction(X=X_test, w_down=w_down, w_up=w_up)
            Y_predict_test = (VI_test > threshold_opt)
            F1_score_test = f1_score(Y_predict_test, Y_test)
            F1_score_test = np.round(F1_score_test, 4)
            messages.addMessage(f"F1_score_test:{F1_score_test}")

        # Export raster with novel fraction indicator
        up = (w_up[0] * arrays["Red"] + w_up[1] * arrays["Green"] + w_up[2] * arrays["Blue"])
        down = (w_down[0] * arrays["Red"] + w_down[1] * arrays["Green"] + w_down[2] * arrays["Blue"])
        down = np.where(down == 0, 1, down)
        array_out_fraction_optimised = up / down

        mask_values = (array_out_fraction_optimised >= -5) * (array_out_fraction_optimised <= 5)
        array_out_fraction_optimised = mask_values * array_out_fraction_optimised -\
            (array_out_fraction_optimised < -5).astype("float") +\
            (array_out_fraction_optimised > 5).astype("float")

        array_to_raster(
            array=array_out_fraction_optimised, fileout="Optimised_fraction_indicator",
            blocksize=2**13, raster_pattern=bands["Red"])

        messages.addMessage("13. Raster for optimised fraction indicators was exported")

        # Remove temporary files
        for object_name in objects_to_remove:
            arcpy.management.Delete(object_name)
        messages.addMessage("Temporary files were removed")

        return
