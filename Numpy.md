---
layout: page
title: Numpy
permalink: /numpy/
nav_order: 10
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

[Official Documentation](http://scipy-lectures.org/intro/numpy/array_object.html#indexing-and-slicing){: .btn .fs-5 .mb-4 .mb-md-0 }

Numpy and Scipy are two resources to compute a variety of functions on matrices. Scipy is built on top of numpy and has a larger codebase of modules which we can utilize

![](/images/numpy_axes.png)

# Images and Arrays

## Image Operations

### Importing Images 

In the below code we input an image and convert it into an array. \
Shape of an array is just it's size

```python
im = array(Image.open('empire.jpg'))
print im.shape, im.dtype

```
The output would look lik this:
```python
(800, 569, 3) uint8 (RGB image)
```
### Converting image to Greyscale

This uses an extra library called Python Pillow

```python
from PIL import Image, ImageOps
im = array(Image.open('empire.jpg').convert('L'),'f')
print im.shape, im.dtype

```

### Plotting an image

```python
img = np.array(Image.open('House2.jpg'))
plt.figure(figsize=(8,8))
plt.imshow(img)
plt.show

```



## Array Functions and Operations

## Array Nomenclature

![](/images/3D_matrix_nomenclature.png)

<b> It's important to realise that we only care about shapes of a matrix and our computation revolves around the shape tuple (1,2,3) irrespective of which is row or column. Develop a generalized version of matrix definitions!!

An image can have shape as (640,540,3). Here we need to think in the way that there are 640 rows and 540 columns and 3 RGB channels. Therefore, rows, columns, pages don't matter much. Just think in terms of shapes.

### sum() function in 1D

```python
import numpy as np 
arr = [20, 2, .2, 10, 4]  
   
print("\nSum of arr : ", np.sum(arr)) 
   
print("Sum of arr(uint8) : ", np.sum(arr, dtype = np.uint8)) 
print("Sum of arr(float32) : ", np.sum(arr, dtype = np.float32))
```
Output:

```python
Sum of arr :  36.2
Sum of arr(uint8) :  36
Sum of arr(float32) :  36.2
```
In 1D it just computes the sum of all elements in the array.
It can also do type conversion on the go.

We can extend this same logic to 2D, there too it calculates the sum of all matrix elements


### sum() in 2D along axes

Axis along which we want to calculate the sum value. Otherwise, it will consider arr to be flattened(works on all the axis). 

axis = 0 means it calculates sum of all elements in ith column and (i=1)th column..

axis = 1 means it calculates sum of all elements in (j)th column and (j+1)th column..

```python
arr = [[14, 17, 12, 33, 44],   
       [15, 6, 27, 8, 19],  
       [23, 2, 54, 1, 4,]]  
   
print("\nSum of arr : ", np.sum(arr)) 
print("Sum of arr(axis = 0) : ", np.sum(arr, axis = 0)) 
print("Sum of arr(axis = 1) : ", np.sum(arr, axis = 1))
```

Output would be:
```python
Sum of arr :  279
Sum of arr(axis = 0) :  [52 25 93 42 67]
Sum of arr(axis = 1) :  [120  75  84]
```

But notice how the vector of axis = 1 has been transposed to show as a row vector

We change that behaviour by adding a second argument to the sum() function:

```python
print("\nSum of arr (keepdimension is True): \n",
      np.sum(arr, axis = 1, keepdims = True))
```

Output
```python
Sum of arr (keepdimension is True): 
 [[120]
 [ 75]
 [ 84]]
```

### Looping over an Image and Grayscale

We can loop over individual elements in a matrix after knowing the shape of the matrix

The shape of the image is given as a tuple eg. (640, 540, 3)
- the last item of that tuple is the RGB spectrum (3 dimensions per pixel)
- the first two items in the tuple is the actual size of the image

```python
for i in range(img.shape[1]):
    print()

```

In the above code we are looping over the rows. Therefore we are looping 640 times.

#### Method 1 : Consider this method of converting image into greyscale:

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

img = np.array(Image.open('B1.jpg'))
print(img.shape)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        grey_value = 0
        for k in range(img.shape[2]):
            grey_value += img[i,j,k]
        img[i,j,0] = int(grey_value/3)
img2 = img[:,:,1]
plt.figure(figsize=(8,8))
plt.imshow(img2)
plt.show()
```
Also note how we removed the third (extra) dimensions using:

```
img2 = img[:,:,1]
```
This method uses averaging to find grayscale. However a slightly modified version is usually preferred:

#### Method 2: Accounting for Luminance Perception

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

weight = [0.2989, 0.5870, 0.1140]

img = np.array(Image.open('B1.jpg'))
print(img.shape)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        grey_value = 0
        for k in range(len(weight)):
            grey_value += (img[i,j,k]*weight[k])
        img[i,j,0] = int(grey_value)
img2 = img[:,:,1]
plt.figure(figsize=(8,8))
plt.imshow(img2, cmap=plt.get_cmap("gray"))
plt.show()
```

#### Method 3: Simpler code using numpy.mean

```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
color_img = np.array(Image.open('B1.jpg')) / 255
img = np.mean(color_img, axis=2)
plt.figure(figsize=(8,8))
plt.imshow(img, cmap=plt.get_cmap("gray"))
plt.show()
```

# Built-in Numpy functions

## Difference between dot, matmul, and *

![](/images/np.dot.png)

## Plotting a pixel-wise histogram

```python
img = np.array(Image.open('emma_stone.jpg'))
img_flat = img.flatten()
plt.hist(img_flat, bins=200, range=[0, 256])
plt.title("Number of pixels in each intensity value")
plt.xlabel("Intensity")
plt.ylabel("Number of pixels")
plt.show()
```

## Reshaping Arrays

```python
x = np.arange(4).reshape((2,2))
x
>>array([[0, 1],
       [2, 3]])
```

## Transpose of a matrix

Simple transpose is done using the matrix.transpose() or matrix.T method (both are same). One of them is showed below:

```python
# (refer matrix x in above example)
np.transpose(x)
array([[0, 2],
       [1, 3]])
```

However the transpose function takes more arguments and this is important for 3D matrices.

<i>Note that if a 3D matrix say 'A' has shape (1,2,3), the result of transpose without specifying any extra argument will be (3,2,1)

```
x = np.ones((1, 2, 3))
np.transpose(x, (1, 0, 2)).shape
>>(2, 1, 3)
```
<i>Note. While declaring array as in np.ones(1,2,3). This can be interpreted in two ways:

- If we are printing the array in terminal we will read it as: there are 1 pages, 2 rows and 3 columns
- If it's an image, the shape will be 1 row, 2 coulmns and 3 will be for 3 RGB channels

<b> It's important to realise that we only care about shapes of a matrix and our computation revolves around the shape tuple (1,2,3) irrespective of which is row or column. Develop a generalized version of matrix definitions!!

However, we will access each row/column starting from 0 as x[0,0,0] or x[1,1,1]. 

The second argument stands for the axes parameter. Axes are numbered as 0,1,2

i.e. default configuration of axes is (0,1,2) for a 3D array and (0,1) for a 2D array

Therefore if we specify <np.transpose(x,(1,0,2))> we're saying that we want the first two shapes interchanged.

Remember that first two shapes are pages and rows. Hence, those two will interchange.

## Padding of Matrices

Padding is used to ensure overall image size does not reduce while run filters/convulutions on it

```python
import numpy as np

x = np.ones(3)
y = np.pad(x, pad_width=1)
y

# Expected result
# array([0., 1., 1., 1., 0.])
```

## newaxis method

ref: [newaxis](http://scipy-lectures.org/intro/numpy/array_object.html#indexing-and-slicing)

This method can be used to convert a row vector to a column vector and at the same time add another dimension as shown below:

```python
a = np.array([0,1,2])
print(a.shape)
```

Output: (3,)

Now lets do the newaxis modification:

```python
c = (a[:, np.newaxis])
print(c)
print(c.shape)
```

Output:
```python
[[0]
 [1]
 [2]]

(3, 1)
```
Therefore we can see that the vector has been rotated and another dimension has been added to the shape tuple

## einsum 

Refer to this document: [einsum](https://ajcr.net/Basic-guide-to-einsum/)

## Stacking rows using vstack

We can use this function to stack rows onto an exiting numpy array.

```python
in_arr1 = geek.array([ 1, 2, 3] )
  
in_arr2 = geek.array([ 4, 5, 6] )
  
# Stacking the two arrays vertically
out_arr = geek.vstack((in_arr1, in_arr2))
print (out_arr)
```

Practically we can use this in a specific case. If we don't know the number of rows we will be adding to a numpy array:

- We will define the array as a 0 row array
- We then add rows as we progress using the vstack function

```python
word_array = np.array([]).reshape(0, maxlength)
    for message in messages:
        word_count = np.zeros((1, word_num))
        for word in word_list:
            if word  == "yes":
                word_count[0, word_dictionary[word]] += 1
        word_array = np.vstack([word_array, word_count])
    return word_array
```

## Saving a numpy matrix in a text file

```python
np.savetxt('./output/p06_sample_train_matrix', train_matrix[:100,:])
```