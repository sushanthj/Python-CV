---
layout: page
title: CV Edge Detection and Filters
permalink: /CV Concepts/
nav_order: 11
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

[Youtube Lectures](https://www.youtube.com/playlist?list=PLmyoWnoyCKo8epWKGHAm4m_SyzoYhslk5){: .btn .fs-5 .mb-4 .mb-md-0 } [Reference Notes](https://www.crcv.ucf.edu/courses/cap5415-fall-2014/){: .btn .fs-5 .mb-4 .mb-md-0 }


The terms convolution is used in a very loose manner and when people say convolution, they mostly mean correlation. However, in our code, we will follow strictly the definition of convolution by flipping the kernel accordingly.

<i> Note that if we have a symmetric kernel, then convolution and correlation would be the same

Also, people include the aspects of filtering (i.e taking a kernel and running it over an image) in the overarching term 'Convolution'.

# Basics

## Image - Kernel - Padding Relations

![](/images/padding_formula.png)

## Convultion

refer: [Conv](http://www.songho.ca/dsp/convolution/convolution.html)
refer: [Conv vs Cross Conv](https://glassboxmedicine.com/2019/07/26/convolution-vs-cross-correlation/)

## Averaging Filter

Here we use a simple np.ones((3,3)) kernel and do convolution calculate the sum of all neighbouring pixels.
After that, we divide the sum by 9 (as 9 elements in the kernel)

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

def convolve2D(image, kernel, padding=0, strides=1):
    # Flipping
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.pad(image, pad_width=padding)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        mat_mul = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape])
                        output[x, y] = (mat_mul.sum()/9)

                except:
                    break

    return output
```
Note that the image we pass is grayscale: 'np.mean(color_img, axis=2)

Also, we are padding the image correctly so that the loop can start at image[0,0] and still not face any issues. Note that the * operator is a different form of multiplication than matmul. Try to understand how the loop works at the boundary conditions

## Sobel Edge detector

- Here we apply two convolutions to get fx and fy. i.e. the kernels 1 and 2 are designed in such a way that they give the central difference along x-axis or y-axis.

- This central difference is essential the derivative of the matrix in x and y directions

- Finally we combine the two fx and fy matrices to get the final resultant matrix

- This resultant matrix has only edges as seen in the picture below.

- Optionally we can also add a threshold to this resultant matrix to keep some edges and discard others

![](/images/sobel_diag.png)
![](/images/B1.jpg)
![](/images/sobel_output.png)

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

def Sobel2D(image, kernel1, kernel2, padding=0, strides=1):
    # Flipping
    kernel1 = np.flipud(np.fliplr(kernel1))
    kernel2 = np.flipud(np.fliplr(kernel2))

    # Gather Shapes of Kernel
    xKernShape = kernel1.shape[0]
    yKernShape = kernel1.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output_x = np.zeros((xOutput, yOutput))
    output_y = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.pad(image, pad_width=padding)
    else:
        imagePadded = image

    # Iterate to get fx (kernel1 is designed to yield differential in row-wise manner, lookup the kernel)
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        mat_mul = (kernel1 * imagePadded[x: x + xKernShape, y: y + yKernShape])
                        output_x[x, y] = (mat_mul.sum())

                except:
                    break
    
    # Iterate to get fy (kernel2 is designed to yield differential in column-wise manner, lookup the kernel)
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        mat_mul = (kernel2 * imagePadded[x: x + xKernShape, y: y + yKernShape])
                        output_y[x, y] = (mat_mul.sum())

                except:
                    break

    #find the resultant vector's magnitude
    resultant_mag = np.sqrt(np.square(output_x) + np.square(output_y))
    
    #Normalize output to be between 0 and 255
    #resultant_mag *= 255/resultant_mag.max()
    return resultant_mag

if __name__ == '__main__':
    color_img = np.array(Image.open('B1.jpg'))
    img = np.mean(color_img, axis=2)
    ker1 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    ker2 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    pad = 1

    new_img = Sobel2D(img, ker1, ker2)
    plt.figure(figsize=(8,8))
    plt.imshow(new_img, cmap=plt.get_cmap("gray"))
    plt.show()
```

## Gaussian Matrix

To map the gaussian distribution onto a kernel, we'll use a function which generates gaussian numbers as a distribution

```python
x, y = np.meshgrid(np.linspace(-1,1,3), np.linspace(-1,1,3))
print(x)
print(y)
dst = np.sqrt(x*x+y*y)
  
#Intializing sigma and muu
sigma = 1
muu = 0.000
  
#Calculating Gaussian array
gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
  
print("2D Gaussian array :\n")
print(gauss)
```

Note here that x and y are two different matrices and are shown below in the following order:

```
x:

[[-1.  0.  1.]
 [-1.  0.  1.]
 [-1.  0.  1.]]

y:

[[-1. -1. -1.]
 [ 0.  0.  0.]
 [ 1.  1.  1.]]
 ```
![](/images/2D_gaussian.jpeg)

## Gaussian Filtering

Using the above gaussian matrix (the final output) as the kernel we do convolutions on the image

```python
from avg_kernel import convolve2D
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

def gaussian(sig):
    ker_size = (2*sig)+1
    x, y = np.meshgrid(np.linspace(-(ker_size // 2),(ker_size // 2), ker_size), np.linspace(-(ker_size // 2),(ker_size // 2), ker_size)) 
    dst = np.sqrt(x*x+y*y)

    #Intializing sigma and muu
    sigma = 1
    muu = 0.000
  
    #Calculating Gaussian array
    gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
    print(gauss)

    return gauss

def convolve(image, kernel, padding=1, strides=1):
    # Flipping
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.pad(image, pad_width=padding)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        mat_mul = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape])
                        output[x, y] = (mat_mul.sum())

                except:
                    break

    return output

if __name__ == '__main__':
    color_img = np.array(Image.open('B1.jpg'))
    img = np.mean(color_img, axis=2)
    ker = gaussian(1)
    pad = 1

    new_img = convolve(img, ker)
    plt.figure(figsize=(8,8))
    plt.imshow(new_img, cmap=plt.get_cmap("gray"))
    plt.show()
```
The final smoothened image with a 3x3 kernel and SD = 1 is shown below:

![](/images/gauss_filter.png)

## Marr-Hildreth Edge Detector

Method 1
- Find the Laplacian of Gaussian matrix
- Convolve with image
- Find zero crossings and evaluate the slope
- Apply threshold to slope

The first two steps are solved in below code
```python
from avg_kernel import convolve2D
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

def lap_of_gauss(sig):
    ker_size = (2*sig)+1
    x, y = np.meshgrid(np.linspace(-(ker_size // 2),(ker_size // 2), ker_size), np.linspace(-(ker_size // 2),(ker_size // 2), ker_size)) 
    dst = np.sqrt(x*x+y*y)

    #Intializing sigma and muu
    muu = 0.000
  
    #Calculating Gaussian array
    p3 = np.exp(-( (dst-muu)**2 / ( 2.0 * sig**2 ) ) )
    print(p3)

    p2 = (2-((dst**2)/(sig**2)))

    p1 = -(1/(((2*3.14)**0.5)*(sig**3)))

    l_of_g = (p1*(p2*p3))
    print(l_of_g)

    return l_of_g

def convolve(image, kernel, padding=1, strides=1):
    # Flipping
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    conv_out = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.pad(image, pad_width=padding)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        mat_mul = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape])
                        conv_out[x, y] = (mat_mul.sum())

                except:
                    break

    return conv_out

if __name__ == '__main__':
    color_img = np.array(Image.open('B1.jpg'))
    img = np.mean(color_img, axis=2)
    ker = lap_of_gauss(1)
    pad = 1

    new_img = convolve(img, ker)
    plt.figure(figsize=(8,8))
    plt.imshow(new_img, cmap=plt.get_cmap("gray"))
    plt.show()
```

Method 2
- Find gaussian matrix
- Convolve gaussian matrix with image
- Find gradient(first derivative) and Double gradient(second derivative) in both x,y axes

The below code shows how to find gradient using the 1D numpy.gradient() function

```python
gradients = numpy.gradient(img)
x_grad = gradients[0]
y_grad = gradients[1]

```
