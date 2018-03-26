import matplotlib.pyplot as plt
import numpy as np
import gdal

from skimage import transform as tf
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float, img_as_ubyte
from  skimage import exposure

image = gdal.Open('image.tif')

def enhance(img):
    enhanced_bands = list()
    for b in range(img.shape[-1]):
        enhanced_bands.append(exposure.equalize_adapthist(img[:, :, b]))
    return np.dstack(enhanced_bands)


img_red_band = image.GetRasterBand(1).ReadAsArray().astype(float)
img_green_band = image.GetRasterBand(2).ReadAsArray().astype(float)
img_blue_band = image.GetRasterBand(3).ReadAsArray().astype(float)

print(img_red_band.shape)
new_array = np.array([])
array_as_list_img_red = []
flag = 1
for i in range(1, np.shape(img_red_band)[0]):
    list_for_row_creation = []
    for j in range(1, np.shape(img_red_band)[1]):
        temp_array = img_red_band[i-1:i+2, j-1:j+2]
        relative_mean = np.mean(temp_array)
        window_variance = np.var(temp_array)
        list_for_row_creation.append(window_variance)
    array_as_list_img_red.append(list_for_row_creation)

array_as_list_img_green = []
for i in range(1, np.shape(img_green_band)[0]):
    list_for_row_creation = []
    for j in range(1, np.shape(img_green_band)[1]):
        temp_array = img_green_band[i-1:i+2, j-1:j+2]
        relative_mean = np.mean(temp_array)
        window_variance = np.var(temp_array)
        list_for_row_creation.append(window_variance)
    array_as_list_img_green.append(list_for_row_creation)
#print(len(array_as_list_img))

array_as_list_img_blue = []
for i in range(1, np.shape(img_blue_band)[0]):
    list_for_row_creation = []
    for j in range(1, np.shape(img_blue_band)[1]):
        temp_array = img_blue_band[i-1:i+2, j-1:j+2]
        relative_mean = np.mean(temp_array)
        window_variance = np.var(temp_array)
        list_for_row_creation.append(window_variance)
    array_as_list_img_blue.append(list_for_row_creation)
#print(len(array_as_list_img))

red_band_img = np.array([xi for xi in array_as_list_img_red])
green_band_img = np.array([xi for xi in array_as_list_img_green])
blue_band_img = np.array([xi for xi in array_as_list_img_blue])

RGB_img = np.dstack((red_band_img / np.amax(red_band_img),green_band_img / np.amax(green_band_img), blue_band_img / np.amax(blue_band_img)))
#print(red_band_img.shape)
#print(red_band_img)
im1 = plt.imshow(enhance(RGB_img) , interpolation='nearest')
plt.show()
#image = (np.array(gdal.Open('image.tif').ReadAsArray())).astype(float)
#print(image)
print(img_red_band)
#image = image.transpose(1, 2, 0)
'''
img_red_band = image.GetRasterBand(1).ReadAsArray().astype(float)
img_green_band = image.GetRasterBand(2).ReadAsArray().astype(float)
img_blue_band = image.GetRasterBand(3).ReadAsArray().astype(float)
print(img_red_band)
'''
#print(image)
#
img_warp= img_as_float(image[::2, ::2])
#tform = tf.AffineTransform(scale=(1.0, 1.0), rotation=0.25)
#img_warp = tf.warp(img, tform)

segments_fz = felzenszwalb(img_warp, scale=100, sigma=0.5, min_size=50)
segments_slic = slic(img_warp, n_segments=250, compactness=10, sigma=1)
segments_quick = quickshift(img_warp, kernel_size=5, max_dist=5, ratio=0.001)
gradient = sobel(rgb2gray(img_warp))
segments_watershed = watershed(gradient, markers=250, compactness=0.001)

print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))
print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
print('Quickshift number of segments: {}'.format(len(np.unique(segments_quick))))

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

ax[0, 0].imshow(mark_boundaries(img_warp, segments_fz))
ax[0, 0].set_title("Felzenszwalbs's method")
ax[0, 1].imshow(mark_boundaries(img_warp, segments_slic))
ax[0, 1].set_title('SLIC')
ax[1, 0].imshow(mark_boundaries(img_warp, segments_quick))
ax[1, 0].set_title('Quickshift')
ax[1, 1].imshow(mark_boundaries(img_warp, segments_watershed))
ax[1, 1].set_title('Compact watershed')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()
