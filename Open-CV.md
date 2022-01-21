---
layout: page
title: OpenCV
permalink: /OpenCV/
nav_order: 12
---

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

# Before you Begin
{: .fs-9 }

[Reference](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html){: .btn .fs-5 .mb-4 .mb-md-0 }

Many numpy functions will still be used alongside OpenCV as it augments the capabilities of OpenCV

## Note: The axes convention for numpy and opencv are different in some instances

For basic image ops:
- numpy has [width, height, depth] as the three axes
- opencv has [height, width, depth] as the three axes

However, for operations such as cv2.rectangle, i.e.

```python
image = cv2.rectangle(image, start_point, end_point, color, thickness)
# the start point is (x,y) i.e. same as numpy
```

But, if we're extracting an ROI (see section below), then we do:
```ball = img[280:340, 330:390]```

here the 280:340 = height and 330:390 = width


# Basic Image Operations

## Read Image, Overlap Image, Show Image and Create New Image

```python
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('walk_1.jpg')
img2 = cv.imread('walk_2.jpg')

img3 = cv.add(img, img2)
cv.imshow('img3', img3)
cv.waitKey(0)
cv.destroyAllWindows()

cv2.imwrite("opencv-threshold-example.jpg", img3)
```

## Split RGB Channels

```python
b,g,r = cv.split(img)
img = cv.merge((b,g,r))

#or we can also use numpy and do faster/more efficiently than cv.split

b = img[:,:,0]

#to set all red pixels to zero

img[:,:,2] = 0
```

## Extracting Region of Interest (ROI)

The axes of an image by default is saved as:

```python
h,w,d = img.shape()
```

In the above height, width, and depth: 
- depth is saved as it normally would be
- height is defined as top of image is zero and bottom of image is max height /
(opposite of the normal y-axis)
- width is defined normally, from left to right like 'x-axis'

```python
ball = img[280:340, 330:390]
img[273:333, 100:160] = ball
```

## Grayscale and Thresholding

There are two common color spaces used in OpenCV\
- cv.COLOR_BGR2GRAY
- cv.COLOR_BGR2HSV

Thresholding sSyntax is given as\
<b>cv.threshold(source, thresholdValue, maxVal, thresholdingTechnique)

Few thresholding techniques are:

- **cv.THRESH_BINARY**: If pixel intensity is greater than the set threshold, value set to 255, else set to 0 (black)
- **cv.THRESH_BINARY_INV**: Inverted or Opposite case of cv.THRESH_BINARY
- **cv.THRESH_TRUNC**: If pixel intensity value is greater than threshold, it is truncated to the threshold. The pixel values are set to be the same as the threshold. All other values remain the same
- **cv.THRESH_TOZERO**: Pixel intensity is set to 0, for all the pixels intensity, less than the threshold value
- **cv.THRESH_TOZERO_INV**: Inverted or Opposite case of cv.THRESH_TOZERO

**The Thresholding will generate two outputs. That's why we'll save the output of cv.THRESH in two images**

```python

# Make image grayscale before thresholding ALWAYS

gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Or make grayscale during image read itself

gray_img = cv.imread("threshold.png", cv.IMREAD_GRAYSCALE)

# Apply Threshold

ret, thresh1 = cv.threshold(grau_img, 120, 255, cv.THRESH_BINARY)

cv.imshow('Binary Threshold', thresh1)
```
## Adaptive Thresholding ( Also nice way to print multiple images)

**Syntax : cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst])**

- src – Source 8-bit single-channel image.
- dst – Destination image of the same size and the same type as src .
- maxValue – Non-zero value assigned to the pixels for which the condition is satisfied. See the details below.
- adaptiveMethod – Adaptive thresholding algorithm to use, ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C . See the details below.
- thresholdType – Thresholding type that must be either THRESH_BINARY or THRESH_BINARY_INV .
- blockSize – Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
- C – Constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be zero or negative as well.

```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('sudoku.png',0)
img = cv.medianBlur(img,5)

ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
```
______________________________________________________________________________________________

## Creating custom window sizes and putting text on images

The below code will also take user input and executes custom commands like an if-else statement

```python
h1, w1, d1 = img.shape
        
        cv2.putText(img, img_orig_path, (20,20) , font, fontScale, fontColor, lineType) 
        img = cv2.rectangle(img, (x1,y1), (x2,y2),(36,255,120), 3)
        cv2.namedWindow("display_image",cv2.WINDOW_NORMAL)
        cv2.resizeWindow("display_image",1000,1000)
        cv2.imshow("display_image", img)
        k1 = cv2.waitKey(0) & 0xFF
        if k1 == ord('q'):
            cv2.destroyAllWindows() 
            print("> User exit request")
            break
        elif k1 == ord('p'):
            print('okay')
        else:    
            pass
```


## Bitwise operations

```python
import numpy as np
import cv2 as cv

# Load two images
img1 = cv.imread('messi5.jpg')
img2 = cv.imread('opencv-logo-white.png')

# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols]

# Now create a mask of logo and create its inverse mask also
img2gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)


# Now black-out the area of logo in ROI
img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv)


# Take only region of logo from logo image.
img2_fg = cv.bitwise_and(img2,img2,mask = mask)


# Put logo in ROI and modify the main image
dst = cv.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst

cv.imshow('res',img1)
cv.imshow('img1_bg', img1_bg)
cv.imshow('img2_fg',img2_fg)
cv.imshow('mask', mask)
cv.imshow('mask_inv', mask_inv)
cv.waitKey(0)
cv.destroyAllWindows()
```

The output will be as follows:
![](/images/opencv1.png)

## Masking an Image

- Firstly the mask has to be same size as image
- Secondly, the mask needs a white region where we'll be performing our bitwise ops

```python
# zero mask unwanted regions in the image
mask = np.zeros([1200,1328], dtype="uint8")
mask = cv2.rectangle(mask,(130,300),(1200,1020),(255,255,255),-1)
depth_img = cv2.bitwise_and(depth_img, depth_img, mask=mask)
```

## Resizing an Image

```python
#Default interpolation method is cv.INTER_LINEAR which is faster than cv.INTER_CUBIC

img = cv.imread('messi5.jpg')
res = cv.resize(img,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)
```

## Translating and Rotating an Image

**Syntax: cv2.warpAffine(src, M, dsize, dst, flags, borderMode, borderValue)**

The parameter M is the transformation matrix and this matrix dictates whether we translate or rotate the image or any of the other several operations

- src: input image
- dst: output image that has the size dsize and the same type as src.
- M: transformation matrix.
- dsize: size of the output image. eg. cv.warpAffine((100,100))
- flags: combination of interpolation methods (see resize() ) and the optional flag
- WARP_INVERSE_MAP that means that M is the inverse transformation (dst->src).
- borderMode: pixel extrapolation method; when borderMode=BORDER_TRANSPARENT, it means that the pixels in the destination image corresponding to the “outliers” in the source image are not modified by the function.
- borderValue: value used in case of a constant border; by default, it is 0.

### Translation

The below example shows x_translation of 150 pixels and y_translation of 50 pixels
```python
img = cv.imread('messi5.jpg',0)
rows,cols = img.shape
M = np.float32([[1,0,100],[0,1,50]])
dst = cv.warpAffine(img,M,(cols,rows))
cv.imshow('img',dst)
cv.waitKey(0)
cv.destroyAllWindows()
```

### Rotation

For rotation we manually find the correct rotation matrix using an inbuilt function of OpenCV

![](/images/opencv2.png)

```python
img = cv.imread('messi5.jpg',0)
rows,cols = img.shape
# cols-1 and rows-1 are the coordinate limits.
M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
dst = cv.warpAffine(img,M,(cols,rows))
```

## Affine Transformation vs Perspective Transformation

In Affine transformation all parallel liles will remain parallel after transformation.

This also needs 3 points to be mapped from input to output as a reference to build the transformation matrix

```python
img = cv.imread('drawing.png')
rows,cols,ch = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

M = cv.getAffineTransform(pts1,pts2)
dst = cv.warpAffine(img,M,(cols,rows))
```

In Perspective transformation all straight lines will remain straight even after transformation

This needs 4 point mapping from input to output image to build transformation matrix

```python
import numpy as np
import cv2 as cv

img = cv.imread('sudoku.png')
rows,cols,ch = img.shape

pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv.getPerspectiveTransform(pts1,pts2)
dst = cv.warpPerspective(img,M,(300,300))
```

______________________________________________________________________________________________________


# Kernel Operations

## Averaging filter

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('opencv_logo.png')

kernel = np.ones((5,5),np.float32)/25
dst = cv.filter2D(img,-1,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
```

## Gaussian Blur

```python

img = cv.imread('opencv-logo-white.png')
img1 = cv.blur(img,(5,5)) #Normal averging filter

img2 = cv.GaussianBlur(img,(5,5))
```

## Perspective Transforms and Homography

In homography we can:
1. Match the FOV of one camera to another camera (which is slighly offset from the first)
2. Warp an image to remove distorion effects

We achieve this by first finding a homography matrix. The basic backend is shown below:

![](/images/homography.png)

The basic code to find the homography matrix and to do the actual perspective transform uses three OpenCV functions:
- ```cv2.findHomography(pts_dst, pts_src)```
- ```cv2.warpPerspective(im_dst, h, (im_src.shape[1],im_src.shape[0]))```
- ```transformed_points = cv2.perspectiveTransform(test_points,h)```

In the above options:

h = transformation matrix \
im_dst = destination image \
im_src = source image \
pts_dst = keypoints on the destination image \
pts_src = keypoints on the source image

**The destination image will be the image which will be transformed to match the source image**


```python
from cgi import test
import os
import cv2
import numpy as np
from datetime import datetime
from numpy.lib.npyio import load

def compute_transformation(src_path, dst_path, out_path="/home/sush/TS/misc/test_scripts/"):
    # Read source image.
    im_src = cv2.imread(src_path)
    # Four corners of the book in source image
    # pts_idx = [1,2,3,4,5,6,13,7,8,9,10,11,12]
    pts_src = np.array([[24,240],[120,223],[252,224],[413,221],[315,332],[367,413],[438,467],[440,410],[365,474],[36,429],[36,499],[27,381]])

    # Read destination image.
    im_dst = cv2.imread(dst_path)
    # Four corners of the book in destination image.
    pts_dst = np.array([[5,129],[115,107],[261,110],[436,106],[327,285],[382,418],[465,516],[465,418],[382,515],[19,438],[12,551],[10,352]])

    # Calculate Homography
    h, status = cv2.findHomography(pts_dst, pts_src)
    #h, status = cv2.getPerspectiveTransform(pts_dst, pts_src)

    test_points = np.array([[328,218],[423,333]])
    test_points = np.float32(test_points).reshape(-1,1,2)

    # Warp destination image to source image based on homography
    im_out = cv2.warpPerspective(im_dst, h, (im_src.shape[1],im_src.shape[0]))
    transformed_points = cv2.perspectiveTransform(test_points,h)
    print("transfored points are \n",transformed_points)

    # Display images
    cv2.imshow("Source Image", im_src)
    cv2.imshow("Destination Image", im_dst)
    cv2.imshow("Warped Source Image", im_out)


    #cv2.imwrite(out_path + "warped_img_{}_pts.png".format(pts_src.shape[0]), im_out)
    np.savetxt(out_path + "trans_matrix_{}_pts.txt".format(pts_src.shape[0]),h, fmt='%s')
    print("h : ", h)

    cv2.waitKey(0)


if __name__ == '__main__' :

    dst_path = "/home/sush/TS/depth_testing/validation/camtop_raw/0.png"
    src_path = "/home/sush/TS/depth_testing/validation/depth_raw/0.png"
    compute_transformation(src_path,dst_path)
```

### Points to Note on perspective transform opencv functions:
1. cv2.warpPerspective is used to transform the whole image and takes ~50ms on a Xavier NX (as of 2022). Therefore it must be used sparingly to prevent the pipeline from slowing down and the ros_node from publishing at the low frequency

2. cv2.perspectiveTransform works only on points in the image (image vectors), and is therefore much faster

*if the requirement is to map object detection boxes in one image onto another, then simply take the box corners and use perspectiveTransform to save computation time*
