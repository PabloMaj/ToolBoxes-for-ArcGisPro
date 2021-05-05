# -*- coding: utf-8 -*-

import arcpy
import numpy as np
import os
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.measure import compare_ssim
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
import cv2
from scipy.ndimage import sobel
import pywt
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy

def check_channels_in_array(arr=None):
    channels = arr.shape[0]
    if channels > 3:
        arr = arr[:3, :, :]
    return arr

def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def data_preparing(in_raster_ref_name=None, in_raster_compar_name=None, ROI_name=None):

    if ROI_name == "":
        in_raster_ref = arcpy.Raster(in_raster_ref_name)
        in_raster_compar = arcpy.Raster(in_raster_compar_name)

    else:
        # clip rasters
        arcpy.management.Clip(
            in_raster=in_raster_ref_name, out_raster="Orthomosaic_reference",
            in_template_dataset=ROI_name, nodata_value=0,
            clipping_geometry="ClippingGeometry")

        arcpy.management.Clip(
            in_raster=in_raster_compar_name, out_raster="Orthomosaic_for_comparison",
            in_template_dataset=ROI_name, nodata_value=0,
            clipping_geometry="ClippingGeometry")

        in_raster_ref = arcpy.Raster("Orthomosaic_reference")
        in_raster_compar = arcpy.Raster("Orthomosaic_for_comparison")

    # left corner of rasters
    in_raster_ref_left = arcpy.Point(in_raster_ref.extent.XMin, in_raster_ref.extent.YMin)
    in_raster_compar_left = arcpy.Point(in_raster_compar.extent.XMin, in_raster_compar.extent.YMin)

    # rasters to numpy arrays
    array_ref = arcpy.RasterToNumPyArray(
        in_raster=in_raster_ref, lower_left_corner=in_raster_ref_left, nodata_to_value=0)
    array_compar = arcpy.RasterToNumPyArray(
        in_raster=in_raster_compar, lower_left_corner=in_raster_compar_left, nodata_to_value=0)

    # check number of channels and modify if necessary
    array_ref = check_channels_in_array(array_ref)
    array_compar = check_channels_in_array(array_compar)

    print(array_ref.shape)
    print(array_compar.shape)

    # change the order of the channels
    array_ref = np.moveaxis(array_ref, 0, -1)
    array_compar = np.moveaxis(array_compar, 0, -1)

    # resize second orthomosaic
    array_compar = cv2.resize(array_compar, (array_ref.shape[1], array_ref.shape[0]))

    # change type of data in numpy array
    array_ref = array_ref.astype("uint8")
    array_compar = array_compar.astype("uint8")

    if "report" not in os.listdir():
        os.mkdir("report\\")
    else:
        pass

    return array_ref, array_compar

def visualization_RGB(array_ref=None, array_compar=None):
    plt.clf()
    plt.figure(figsize=(8, 4), dpi=300)
    plt.subplot(121)
    plt.title("\nReference orthomosaic")
    plt.imshow(array_ref)
    plt.axis("off")
    plt.subplot(122)
    plt.title("\nCompared orthomosaic")
    plt.imshow(array_compar)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig('report\\Visualization_RGB.png', dpi=300)

def histograms_analysis(array_ref=None, array_compar=None):
    plt.clf()
    grey_ref = rgb2gray(array_ref).flatten()
    grey_compar = rgb2gray(array_compar).flatten()
    plt.hist(grey_ref, bins=np.arange(1, 255, 1), density=True, alpha=0.5)
    plt.hist(grey_compar, bins=np.arange(1, 255, 1), density=True, alpha=0.5)
    plt.xlabel("Pixel intensity in grayscale image")
    plt.ylabel("Frequency normalized")
    plt.legend(["Reference\northomosaic", "Compared\northomosaic"])
    plt.title("Histograms for grayscale orthomosaics")
    plt.tight_layout()
    plt.savefig('report\\Analiza_histograms.png', dpi=300)

def SSIM_analysis(array_ref=None, array_compar=None):
    plt.clf()
    S, mssim = compare_ssim(X=array_ref, Y=array_compar, win_size=7, multichannel=True, full=True)
    mssim_averaged = np.mean(mssim, axis=-1)
    min_ = np.min(mssim_averaged)
    max_ = np.max(mssim_averaged)
    mssim_normalized = (mssim_averaged - min_) / (max_ - min_)
    plt.clf()
    plt.imsave('report\\ssim_metric.png', mssim_normalized, cmap="brg", vmin=0, vmax=1)
    plt.imsave("report\\orto_ref.png", array_ref.astype("uint8"))
    background = Image.open("report\\orto_ref.png")
    overlay = Image.open("report\\ssim_metric.png")
    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")
    orto_with_ssim = Image.blend(background, overlay, 0.4)
    plt.clf()
    plt.imshow(mssim_normalized, cmap="brg", vmin=0, vmax=1)
    plt.colorbar(label="Measure of similarity normalized")
    plt.imshow(orto_with_ssim)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig('report\\Analiza_ssim.png', dpi=300)
    os.remove("report\\orto_ref.png")
    os.remove("report\\ssim_metric.png")

def FFT_analysis(array_ref=None, array_compar=None):
    orto_compar = rgb2gray(array_compar)
    orto_ref = rgb2gray(array_ref)
    fft_compar = np.fft.fft2(orto_compar)
    fft_ref = np.fft.fft2(orto_ref)
    fft_compar = np.fft.fftshift(fft_compar)
    fft_ref = np.fft.fftshift(fft_ref)
    fft_compar_abs = np.log(1 + np.abs(fft_compar))
    fft_ref_abs = np.log(1 + np.abs(fft_ref))
    max_ = max([np.max(fft_compar_abs), np.max(fft_ref_abs)])
    fft_compar_abs /= max_
    fft_ref_abs /= max_

    plt.clf()
    plt.figure(figsize=(8, 8), dpi=300)

    plt.subplot(221), plt.imshow(fft_ref_abs, "jet")
    plt.title("Centered Spectrum\nfor reference orthomosaic")
    plt.axis('off')
    plt.colorbar(label="Magnitude normalized", orientation="horizontal", fraction=0.046, pad=0.04)

    plt.subplot(222), plt.imshow(fft_compar_abs, "jet")
    plt.title("Centered Spectrum\nfor compared orthomosaic")
    plt.axis('off')
    plt.colorbar(label="Magnitude normalized", orientation="horizontal", fraction=0.046, pad=0.04)

    plt.subplot(223)
    plt.plot(fft_ref_abs[fft_ref_abs.shape[0] // 2, :], alpha=0.5)
    plt.plot(fft_compar_abs[fft_compar_abs.shape[0] // 2, :], alpha=0.5)
    plt.xticks([])
    plt.xlabel("Position in image")
    plt.ylabel("Magnitude normalized")
    plt.title("Spectrum along the central vertical line")
    plt.legend(["Reference\northomosaic", "Compared\northomosaic"])

    plt.subplot(224)
    plt.plot(fft_ref_abs[:, fft_ref_abs.shape[1] // 2], alpha=0.5)
    plt.plot(fft_compar_abs[:, fft_compar_abs.shape[1] // 2], alpha=0.5)
    plt.xticks([])
    plt.xlabel("Position in image")
    plt.ylabel("Magnitude normalized")
    plt.title("Spectrum along the central horizontal line")
    plt.legend(["Reference\northomosaic", "Compared\northomosaic"])

    plt.tight_layout()
    plt.savefig("report\\Analiza_fft.png")

def EI_analysis(array_ref=None, array_compar=None):
    array_ref_grey = rgb2gray(array_ref)
    array_compar_grey = rgb2gray(array_compar)
    sobel_ref = sobel(array_ref_grey)
    sobel_compar = sobel(array_compar_grey)

    plt.clf()
    plt.figure(figsize=(7, 4), dpi=300)
    plt.subplot(121)
    plt.imshow(sobel_ref, "gray")
    plt.title(f"Sobel edges detection for\nReference orthomosaic\nGradient_sum={np.round(np.mean(np.abs(sobel_ref)), 0)}")
    plt.colorbar(label="Gradient", orientation="horizontal", fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(sobel_compar, "gray")
    plt.title(f"Sobel edges detection for\nCompared orthomosaic\nGradient_sum={np.round(np.mean(np.abs(sobel_compar)), 0)}")
    plt.colorbar(label="Gradient", orientation="horizontal", fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("report\\Analiza_Sobel_edges.png")

    gradient_sum_ref = np.mean(np.abs(sobel_ref))
    gradient_sum_compar = np.mean(np.abs(sobel_compar))

    return gradient_sum_ref, gradient_sum_compar

def calculate_FISH(array=None):
    cA, cH, cV, cD = pywt.wavedec2(array, 'bior4.4', mode='periodization', level=3)
    wavelet_subbands = dict()
    wavelet_subbands["LH"] = cH
    wavelet_subbands["HL"] = cV
    wavelet_subbands["HH"] = cD
    log_energy = dict()
    for id_ in ["LH", "HL", "HH"]:
        log_energy[id_] = [np.log10(1 + np.mean(x**2)) for x in wavelet_subbands[id_]]
    log_energy_level = dict()
    for i in [0, 1, 2]:
        log_energy_level[i] = 0.2 * (log_energy["HL"][i] + log_energy["LH"][i]) / 2 + 0.8 * log_energy["HH"][i]
    FISH = 4 * log_energy_level[0] + 2 * log_energy_level[1] + 1 * log_energy_level[2]

    return FISH

class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Comparison of orthomosaics toolbox"
        self.alias = "comparison of orthomosaics"

        # List of tool classes associated with this toolbox
        self.tools = [Tool]


class Tool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Compare orthomosaics"
        self.description = "The tool compares orthomosaics based on structural similarity (SSIM), + \
            pixel instensity histograms and frequency charts after Fast Fourier Transform (FFT).  "
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""

        in_raster_ref = arcpy.Parameter(
            displayName="Reference orthomosaic",
            name="in_raster_ref",
            datatype="Raster Dataset",
            parameterType="Required",
            direction="Input"
        )

        in_raster_compar = arcpy.Parameter(
            displayName="Orthomosaic for comparison",
            name="in_raster_compar",
            datatype="Raster Dataset",
            parameterType="Required",
            direction="Input"
        )

        ROI = arcpy.Parameter(
            displayName="Region of Interest",
            name="ROI",
            datatype="Feature Class",
            parameterType="Optional",
            direction="Input"
        )

        params = [in_raster_ref, in_raster_compar, ROI]
        return params

    def isLicensed(self):  # optional
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):  # optional
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):  # optional
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""

        # input parameters
        in_raster_ref_name = parameters[0].valueAsText
        in_raster_compar_name = parameters[1].valueAsText
        ROI_name = parameters[2].valueAsText

        # Data preparing
        array_ref, array_compar = data_preparing(
            in_raster_ref_name=in_raster_ref_name, in_raster_compar_name=in_raster_compar_name, ROI_name=ROI_name)

        tasks = {
            "visulization_RGB": 1,
            "histograms": 1,
            "ssim": 0,
            "fft": 1,
            "ei_sobel": 1,
        }

        # Visualization RGB
        if tasks["visulization_RGB"]:
            visualization_RGB(array_ref=array_ref, array_compar=array_compar)

        # Histograms analysis
        if tasks["histograms"]:
            histograms_analysis(array_ref=array_ref, array_compar=array_compar)

        # SSIM analysis
        if tasks["ssim"]:
            SSIM_analysis(array_ref=array_ref, array_compar=array_compar)

        # FFT analysis
        if tasks["fft"]:
            FFT_analysis(array_ref=array_ref, array_compar=array_compar)

        # Edge Intensity (Sobel filter) analysis
        if tasks["ei_sobel"]:
            EI_ref, EI_compar = EI_analysis(array_ref=array_ref, array_compar=array_compar)

        # FISH
        array_ref_grey = rgb2gray(array_ref)
        array_compar_grey = rgb2gray(array_compar)
        FISH_ref = calculate_FISH(array=array_ref_grey)
        FISH_compar = calculate_FISH(array=array_compar_grey)

        # SD
        SD_ref = np.ma.array(array_ref_grey, mask=(array_ref_grey == 0)).std()
        SD_compar = np.ma.array(array_compar_grey, mask=(array_compar_grey == 0)).std()

        # MM
        MM_ref = np.ma.array(array_ref_grey, mask=(array_ref_grey == 0)).mean()
        MM_compar = np.ma.array(array_compar_grey, mask=(array_compar_grey == 0)).mean()

        # SM
        SM_ref = skew(array_ref_grey[array_ref_grey > 0].flatten())
        SM_compar = skew(array_compar_grey[array_compar_grey > 0].flatten())

        # KM
        KM_ref = kurtosis(array_ref_grey[array_ref_grey > 0].flatten())
        KM_compar = kurtosis(array_compar_grey[array_compar_grey > 0].flatten())

        # EM
        EM_ref = shannon_entropy(array_ref_grey[array_ref_grey > 0].flatten())
        EM_compar = shannon_entropy(array_compar_grey[array_compar_grey > 0].flatten())

        # Summary
        summary = open("report\\summary.txt", "w")
        summary.write(f"Reference orthomosaic:\t{in_raster_ref_name}\n")
        summary.write(f"Compared orthomosaic:\t{in_raster_compar_name}\n")
        summary.write(f"Region of interest:\t{ROI_name}\n\n")
        summary.write("metric\treference raster\tcompared raster\n")
        summary.write(f"FISH\t{np.round(FISH_ref, 2)}\t{np.round(FISH_compar, 2)}\n")
        summary.write(f"EI\t{np.round(EI_ref, 2)}\t{np.round(EI_compar, 2)}\n")
        summary.write(f"MM\t{np.round(MM_ref, 2)}\t{np.round(MM_compar, 2)}\n")
        summary.write(f"SD\t{np.round(SD_ref, 2)}\t{np.round(SD_compar, 2)}\n")
        summary.write(f"SM\t{np.round(SM_ref, 2)}\t{np.round(SM_compar, 2)}\n")
        summary.write(f"KM\t{np.round(KM_ref, 2)}\t{np.round(KM_compar, 2)}\n")
        summary.write(f"EM\t{np.round(EM_ref, 2)}\t{np.round(EM_compar, 2)}\n")
        summary.close()

        """
        if ROI_name != "":
            arcpy.management.Delete("Orthomosaic_reference")
            arcpy.management.Delete("Orthomosaic_for_comparison")
        """

        print("Report generation process was successful")

        return 1
