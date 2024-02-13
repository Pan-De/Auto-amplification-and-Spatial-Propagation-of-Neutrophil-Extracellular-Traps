import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from scipy import ndimage
from skimage.measure import block_reduce
import math
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

def block_calc_2D(in_shp, trgt_shp):
    return int(in_shp[0]/trgt_shp[0]), int(in_shp[1]/trgt_shp[1])
    
#img/maximum value of the img
def normalize1(img):
    return (img/np.max(img)).astype(np.float64)
#img/a value
def normalize2(img, val):
    return (img/val).astype(np.float64)


#add colorbar to the color map
def plt_funColorbar(arr, title, size=(10,10)):
    plt.figure(figsize=size)
    plt.imshow(arr)
    plt.title(title)
    plt.colorbar()
    plt.show()
# color bar save
def plt_funColorbarSave(arr, title,quality,FileName):  
    plt.imshow(arr,vmin=0,vmax=1)
    plt.title(title)
    cb=plt.colorbar()
    cb.remove()
    plt.colorbar()
    plt.savefig(FileName, dpi=quality)
    

# load the images
#EGFP
path= r'\Well2_140min_XY1_EGFP_10X.tif'
img= Image.open(path)
img_arr_180 = np.array(img)

path= r'\Well2_20min_XY1_EGFP_10X.tif'
img= Image.open(path)
img_arr_20 = np.array(img)

path= r'\Well2_140min_XY1_EGFP_10X.tif'
img= Image.open(path)
img_arr_140 = np.array(img)

path= r'\Well2_100min_XY1_EGFP_10X.tif'
img= Image.open(path)
img_arr_100 = np.array(img)

path= r'\Well2_60min_XY1_EGFP_10X.tif'
img= Image.open(path)
img_arr_60 = np.array(img)


#DAPI
path= r'\Well2_140min_XY1_DAPI_10X.tif'
img= Image.open(path)
dapi_arr_180 = np.array(img)

path= r'\Well2_140min_XY1_DAPI_10X.tif'
img= Image.open(path)
dapi_arr_140 = np.array(img)

path= r'\Well2_100min_XY1_DAPI_10X.tif'
img= Image.open(path)
dapi_arr_100 = np.array(img)

path= r'\Well2_60min_XY1_DAPI_10X.tif'
img= Image.open(path)
dapi_arr_60 = np.array(img)

path= r'\Well2_20min_XY1_DAPI_10X.tif'
img= Image.open(path)
dapi_arr_20 = np.array(img)

#CY5
path= r'\Well2_140min_XY1_CY5_10X1.tif'
img= Image.open(path)
cy5_arr_180 = np.array(img)

path= r'\Well2_140min_XY1_CY5_10X1.tif'
img= Image.open(path)
cy5_arr_140 = np.array(img)

path= r'\Well2_100min_XY1_CY5_10X1.tif'
img= Image.open(path)
cy5_arr_100 = np.array(img)

path= r'\Well2_60min_XY1_CY5_10X1.tif'
img= Image.open(path)
cy5_arr_60 = np.array(img)

path= r'E:\Alec\pan\Well2_20min_XY1_CY5_10X1.tif'
img= Image.open(path)
cy5_arr_20 = np.array(img)

# Subtraction: middle section to be subtracted out of calculations from dapi stain
# PARAMS:
compress_size = (512, 512) 
gauss_sigma = 10 # smooth DAPI blob for consistency
sub_thresh = 50 
erosion_cycles = 10

def subtraction_array (dapi_arr):
    subtraction = cv2.resize(dapi_arr, dsize=compress_size, interpolation=cv2.INTER_CUBIC)
    subtraction = ndimage.gaussian_filter(subtraction, gauss_sigma)
    subtraction = subtraction/np.max(subtraction)*100
    subtraction[subtraction < sub_thresh] = 0
    subtraction[subtraction >= sub_thresh] = 1
    subtraction = ndimage.binary_dilation(subtraction.astype(np.uint8), iterations=erosion_cycles)
    subtraction = 1 - subtraction
    subtraction = cv2.resize(subtraction.astype(float), dsize=dapi_arr.shape, interpolation=cv2.INTER_CUBIC)
    return subtraction


subtraction_180=subtraction_array(dapi_arr_180)
subtraction_140=subtraction_array(dapi_arr_140)
subtraction_100=subtraction_array(dapi_arr_100)
subtraction_60=subtraction_array(dapi_arr_60)
subtraction_20=subtraction_array(dapi_arr_20)

# Subtract background in EGFP images
sample_area = 100 # length of square to calculate background in
background_percentile = 0.6 # zero out cells below this brightness
s_len = int(sample_area/2)

def egfp_arr(img_arr,subtraction):
    new_arr = np.zeros_like(img_arr)
    for i in range(s_len,img_arr.shape[0], sample_area):
        for j in range(s_len,img_arr.shape[1], sample_area):
            curr_area = img_arr[i-s_len:i+s_len, j-s_len:j+s_len]
            background = np.percentile(curr_area, background_percentile)
            curr_area = curr_area - background
            curr_area[curr_area < 0] = 0
            new_arr[i-s_len:i+s_len, j-s_len:j+s_len] = curr_area
        print(f"Processing {i/img_arr.shape[0]:.2%}", end='\r')
    egfp_arr = np.multiply(new_arr, subtraction)
    return egfp_arr

egfp_arr_180=egfp_arr(img_arr_180,subtraction_180)
egfp_arr_140=egfp_arr(img_arr_140,subtraction_140)
egfp_arr_100=egfp_arr(img_arr_100,subtraction_100)
egfp_arr_60=egfp_arr(img_arr_60,subtraction_60)
egfp_arr_20=egfp_arr(img_arr_20,subtraction_20)

# Remove low-brightness noise and high-brightness dead cells
# PARAMS:
low = 6 # Delete noise 
high = 20 # Delete cell pix

def filter_arr(egfp_arr):
    filtered_arr = np.array(egfp_arr)
    filtered_arr[filtered_arr < low] = 0
    filtered_arr[filtered_arr > high] = 0
    return filtered_arr

filter_arr_180=filter_arr(egfp_arr_180)
filter_arr_140=filter_arr(egfp_arr_140)
filter_arr_100=filter_arr(egfp_arr_100)
filter_arr_60=filter_arr(egfp_arr_60)
filter_arr_20=filter_arr(egfp_arr_20)

# Calculate sum NETs of each square segment
# PARAMS sum NETs
area = (100,100) # Sum score in rectangles with shape
def area_calc(filtered_arr):
    sum_arr = filtered_arr
    sum_arr[sum_arr > 0] = 1
    area_calc = block_reduce(sum_arr, area, np.sum)
    return area_calc

area_calc_180=area_calc(filter_arr_180)
area_calc_140=area_calc(filter_arr_140)
area_calc_100=area_calc(filter_arr_100)
area_calc_60=area_calc(filter_arr_60)
area_calc_20=area_calc(filter_arr_20)

# Calculate NET score of each square segment
def norm_arr(area_calc,cy5_arr):
    norm_arr_cy5 = normalize1(block_reduce(cy5_arr, area, np.sum))
    area_calc=np.divide(normalize2(area_calc,np.max(area_calc_180)),norm_arr_cy5)
    return area_calc

norm_arr_180=norm_arr(area_calc_180,cy5_arr_180)
norm_arr_140=norm_arr(area_calc_140,cy5_arr_140)
norm_arr_100=norm_arr(area_calc_100,cy5_arr_100)
norm_arr_60=norm_arr(area_calc_60,cy5_arr_60)
norm_arr_20=norm_arr(area_calc_20,cy5_arr_20)

# Normalize the NET scores to be between 0 to 1
norm_score_180=normalize2 (norm_arr_180,255)
norm_score_140=normalize2 (norm_arr_140,255)
norm_score_100=normalize2 (norm_arr_100,255)
norm_score_60=normalize2 (norm_arr_60,255)
norm_score_20=normalize2 (norm_arr_20,255)

# show NET scores
fig, (ax,ax2,cax) = plt.subplots(nrows=1, ncols=3, figsize=(10, 10),subplot_kw={'xticks': [], 'yticks': []},gridspec_kw={"width_ratios":[1,1, 0.05]})

im1=ax.imshow(norm_score_180)
ax.set_title('180 min')
im2=ax2.imshow(norm_arr_20)
ax2.set_title('20 min')

ip = InsetPosition(ax2, [1.05,0,0.05,1]) 
cax.set_axes_locator(ip)

fig.colorbar(im1, cax=cax, ax=[ax,ax2])
plt.savefig('B:\delete/180_20.png', dpi=1000)
plt.show()