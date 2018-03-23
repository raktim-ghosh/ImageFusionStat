from osgeo import gdal
import numpy as np
from texttable import Texttable


def get_reshaped_image(img1, img2, band):
    img1_band = img1.GetRasterBand(band).ReadAsArray()
    img2_band = img2.GetRasterBand(band).ReadAsArray()
    shape1 = np.shape(img1_band)
    shape2 = np.shape(img2_band)
    new_dim1 = min(shape1[0], shape2[0])
    new_dim2 = min(shape1[1], shape2[1])
    new_list_img1 = img1_band[:new_dim1, :new_dim2]
    new_list_img1 = new_list_img1.reshape((1, new_dim2 * new_dim1))
    new_list_img1 = new_list_img1.tolist()[0]
    new_list_img2 = img2_band[:new_dim1, :new_dim2]
    new_list_img2 = new_list_img2.reshape((1, new_dim2 * new_dim1))
    new_list_img2 = new_list_img2.tolist()[0]

    return [new_list_img1, new_list_img2]


def get_covariance(img1, img2, band):
    list_of_image = get_reshaped_image(img1, img2, band)
    mean_img1 = np.mean(list_of_image[0])
    mean_img2 = np.mean(list_of_image[1])
    cov = 0
    for i, j in zip(list_of_image[0], list_of_image[1]):
        cov += (i - mean_img1) * (j - mean_img2)
    cov = cov / (len(list_of_image[0]) - 1)

    return cov


def get_correlation_coefficient(img1, img2, band):
    list_of_image = get_reshaped_image(img1, img2, band)
    cfr = 0
    cf = 0
    cv = 0
    for i, j in zip(list_of_image[0], list_of_image[1]):
        cfr += i * j
        cf += i * i
        cv += j * j

    correlation = 2 * cfr / (cf + cv)

    return correlation


def get_universal_img_quality_index(img1, img2, band):
    list_of_image = get_reshaped_image(img1, img2, band)
    uiqi = 4 * get_covariance(img1, img2, band) * np.mean(list_of_image[0]) * np.mean(list_of_image[1]) / ((np.var(list_of_image[0]) + np.var(list_of_image[1])) * (np.mean(list_of_image[0]) ** 2 + np.mean(list_of_image[1]) ** 2))

    return uiqi


def get_relative_mean(img1, img2, band):
    relative_mean = 0
    list_of_image = get_reshaped_image(img1, img2, band)
    if img1 == multispectral_image:
        relative_mean = (1 - np.mean(list_of_image[1]) / np.mean(list_of_image[0])) * 100
    elif img2 == multispectral_image:
        relative_mean = (1 - np.mean(list_of_image[0]) / list_of_image[1]) * 100

    return abs(relative_mean)


def get_probability_distribution_of_image(img1, img2, band):
    list_of_image = get_reshaped_image(img1, img2, band)
    probability_distribution_img1 = {}
    for i in range(len(list_of_image[0])):
        count = 0
        if list_of_image[0][i] not in (key for key in probability_distribution_img1):
            for j in list_of_image[0]:
                if list_of_image[0][i] == j:
                    count += 1
            if list_of_image[0][i] not in (key for key in probability_distribution_img1):
                probability_distribution_img1[list_of_image[0][i]] = count/len(list_of_image[0])

    probability_distribution_img2 = {}
    for i in range(len(list_of_image[1])):
        count = 0
        if list_of_image[1][i] not in (key for key in probability_distribution_img2):
            for j in list_of_image[1]:
                if list_of_image[1][i] == j:
                    count += 1
            if list_of_image[1][i] not in (key for key in probability_distribution_img2):
                probability_distribution_img2[list_of_image[1][i]] = count/len(list_of_image[1])

    return [probability_distribution_img1, probability_distribution_img2]


def get_new_probability_distribution(img1, img2, band):
    list_of_image = get_reshaped_image(img1, img2, band)
    arr_hist_img1, bin_img1 = np.histogram(list_of_image[0], bins=65535, range=(0, 65535), density=True)
    arr_hist_img2, bin_img2 = np.histogram(list_of_image[1], bins=65535, range=(0, 65535), density=True)

    return [[arr_hist_img1, bin_img1], [arr_hist_img2, bin_img2]]


def get_entropy_of_images(img1, img2, band):
    list_of_image = get_reshaped_image(img1, img2, band)
    list_of_distribution = get_new_probability_distribution(img1, img2, band)
    entropy_img1 = 0
    #print(len(list_of_image[0]))
    for i in range(len(list_of_distribution[0][0])):
        if list_of_distribution[0][0][i] != 0.0:
            entropy_img1 += -list_of_distribution[0][1][i]*np.log2(list_of_distribution[0][0][i])
    entropy_img2 = 0
    for i in range(0, len(list_of_distribution[1][0])):
        if list_of_distribution[1][0][i] != 0.0:
            entropy_img2 += -list_of_distribution[1][1][i]*np.log2(list_of_distribution[1][0][i])

    if list_of_image[0] == multispectral_image:
        return entropy_img1
    else:
        return entropy_img2


def get_root_mean_square_error(img1, img2, band):
    rmse = 0
    list_of_image = get_reshaped_image(img1, img2, band)
    for i, j  in zip(list_of_image[0], list_of_image[1]):
        rmse += (i-j)**2
    rmse = (rmse/(len(list_of_image[0])))**(0.5)

    return rmse


def get_power_signal_to_noise_ratio(img1, img2, band):
    psnr = 20 * np.log10(65535.0/get_root_mean_square_error(img1, img2, band))
    return psnr


def perform_analysis(img1, img2, band, fp):
    fp.write("IMAGE = " + img2.GetDescription() + "\n")
    uiqi = get_universal_img_quality_index(img1, img2, band)
    fp.write("IMAGE QUALITY INDEX BAND" + str(uiqi) + "\n")
    covariance = get_covariance(img1, img2, band)
    #print(covariance)
    fp.write("IMAGE COVARIANCE BAND" + str(covariance) + "\n")
    correlation = get_correlation_coefficient(img1, img2, band)
    #print(correlation)
    fp.write("IMAGE CORRELATION BAND" + str(correlation) + "\n")
    entropy = get_entropy_of_images(img1, img2, band)
    #print(entropy)
    fp.write("IMAGE CORRELATION ENTROPY" + str(entropy) + "\n")
    rmse = get_root_mean_square_error(img1, img2, band)
    #print(rmse)
    fp.write("IMAGE RMSE" + str(rmse) + "\n")
    relative_mean = get_relative_mean(img1, img2, band)
    #print(relative_mean)
    fp.write("IMAGE RELATIVE MEAN" + str(relative_mean) + "\n")
    psnr = get_power_signal_to_noise_ratio(img1, img2, band)
    #print(psnr)
    fp.write("IMAGE PSNR MEAN" + str(psnr) + "\n")

    return [uiqi, covariance, correlation, entropy, rmse, relative_mean, psnr]


multispectral_image = gdal.Open(r"E:\Refined_Output\aoi.img")
pca = gdal.Open(r"E:\Refined_Output\pca.img")
brovey = gdal.Open(r"E:\Refined_Output\brovey.img")
ehlers = gdal.Open(r"E:\Refined_Output\ehlers.img")
hpf = gdal.Open(r"E:\Refined_Output\hpf.img")
modified_ihs = gdal.Open(r"E:\Refined_Output\modified_ihs.img")
multiplicative = gdal.Open(r"E:\Refined_Output\multiplicative.img")

list_of_images = [multispectral_image, pca, brovey, ehlers, hpf, modified_ihs, multiplicative]
fp = open('Out.txt', 'w')
list_of_all_output = []

for images in list_of_images:
    if images != multispectral_image:
        for band in range(1,images.RasterCount + 1):
            list_of_all_output.append(perform_analysis(multispectral_image, images, band, fp))

for i in range(2,len(list_of_all_output)):
    if (i-2)%3 == 0:
        table = Texttable()
        table.set_cols_align(["c", "c", "c", "c"])
        table.set_cols_valign(["t", "t", "t", "t"])
        table.add_rows([["MEASURE", "BAND-I", "BAND-II", "BAND-III"],
                        ["UIQI", list_of_all_output[i-2][0], list_of_all_output[i-1][0], list_of_all_output[i][0]],
                        ["COVARIANCE", list_of_all_output[i-2][1], list_of_all_output[i-1][1], list_of_all_output[i][1]],
                        ["CORR_COEFF", list_of_all_output[i-2][2], list_of_all_output[i-1][2], list_of_all_output[i][2]],
                        ["ENTROPY", list_of_all_output[i-2][3], list_of_all_output[i-1][3], list_of_all_output[i][3]],
                        ["RMSE", list_of_all_output[i-2][4], list_of_all_output[i-1][4], list_of_all_output[i][4]],
                        ["RMEAN", list_of_all_output[i-2][5], list_of_all_output[i-1][5], list_of_all_output[i][5]],
                        ["PSNR", list_of_all_output[i-2][6], list_of_all_output[i-1][6], list_of_all_output[i][6]]])

        print(table.draw())
        print("\n\n")





