---
layout: page
title: Objects
permalink: /Objects/
nav_order: 6
---

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

# Difference between <i> 'import module' and 'from module import'

import module only pulls the module as an object and like any other object, it has only attributes

```python
>>> import types
>>> types.FunctionType
<type 'function'>

>>> FunctionType
NameError: There is no variable named 'FunctionType'
```

However, when we call Functiontype specifically, or if we do 'from types import *', the FunctionType gets added into the local namespace. Therefore we can do as below:

```python
>>> from types import FunctionType
>>> FunctionType
<type 'function'>

```

# Classes

## Note on __init__

__init__ methods are optional, but when you define one, you must remember to explicitly call the ancestor's __init__ method (if it defines one).

This is more generally true: whenever a descendant wants to extend the behavior of the ancestor, the descendant method must explicitly call the ancestor method at the proper time, with the proper arguments.

## Class attributes

Lets import a module 'fileinfo', and then lets create an instace of a class present within the module called 'FileInfo'


```python
>>> import fileinfo
>>> f = fileinfo.FileInfo("/music/_singles/kairo.mp3")

>>> f.__class__
<class fileinfo.FileInfo at 010EC204>

>>> f.__doc__
'store file metadata'
```
Every class instance has a built−in attribute, __class__, which is the object's class. 

Java programmers may be familiar with the Class class, which contains methods like getName
and getSuperclass to get metadata information about an object. In Python, this kind of metadata is
available directly on the object itself through attributes like __class__, __name__, and __bases__.

<i> if the variable f were to be in a loop, then the moment the loop ends, the instantiation of the class is also destroyed automatically. Therefore, python has a built-in protection against memory leaks

## Some terminology

Let's take this base class as reference:
```python
class UserDict:
    def __init__(self, dict=None):
        self.x = {}
        if dict is not None: self.update(dict)
    
    def somerand(self):
        pass
```

We know the role of the self.x variable here. But more formally it's called a <b> "data attribute" as it is defined within the __init__ function of the class itself

However, if we instantiate the class elsewhere and assign it to a variable eg. <i = UserDict()>
Then the resulting variable i.somerand() is called a __"class attribute"__ and depend on the particular instantiation of the class -> therefore, they aren't permanent like __data attributes__

<i> Also note the above if statement doesn't have an indented block because only one statement is to be performed after logical evaluation. Therefore, it's basically just a shortcut representation

### Side Note on object identity and object equality

In Java, you determine whether two string variables reference the same physical memory location by using str1 == str2. This is called object identity, and it is written in Python as str1 is str2. 

To compare string values in Java, you would use str1.equals(str2); in Python, you would use str1 == str2.

# Examples on working with classes + built-in methods

```python
import math

class Complex(object):
# Classes Dealing with Complex Numbers in python - Hacker Rank Solution START
    def __init__(self, real, imaginary):
        self.real = real
        self.imaginary = imaginary
        
    def __add__(self, no):
        real = self.real + no.real
        imaginary = self.imaginary + no.imaginary
        return Complex(real,imaginary)
        
    def __sub__(self, no):
        real = self.real - no.real
        imaginary = self.imaginary - no.imaginary
        return Complex(real,imaginary)
        
    def __mul__(self, no):
        real = self.real * no.real - self.imaginary * no.imaginary
        imaginary = self.real * no.imaginary + self.imaginary * no.real
        return Complex(real,imaginary)

    def __truediv__(self, no):
        x = float(no.real ** 2 + no.imaginary ** 2)
        y = self * Complex(no.real, -no.imaginary)
        real = y.real / x
        imaginary = y.imaginary / x
        return Complex(real, imaginary)

    def mod(self):
        real = math.sqrt(self.real ** 2 + self.imaginary ** 2)
        return Complex(real, 0)
# Classes Dealing with Complex Numbers in python - Hacker Rank Solution END

    def __str__(self):
        if self.imaginary == 0:
            result = "%.2f+0.00i" % (self.real)
        elif self.real == 0:
            if self.imaginary >= 0:
                result = "0.00+%.2fi" % (self.imaginary)
            else:
                result = "0.00-%.2fi" % (abs(self.imaginary))
        elif self.imaginary > 0:
            result = "%.2f+%.2fi" % (self.real, self.imaginary)
        else:
            result = "%.2f-%.2fi" % (self.real, abs(self.imaginary))
        return result

if __name__ == '__main__':
    c = map(float, input().split())
    d = map(float, input().split())
    x = Complex(*c)
    y = Complex(*d)
    print(*map(str, [x+y, x-y, x*y, x/y, x.mod(), y.mod()]), sep='\n')
```

In the above code notice clearly how:
- Built in functions are used and called (these functions don't have to be called outright, the `<+>` operator or `<*>` operator is called according to the built-in function)

- The return from each function is saved in each individual class instance itself

- We can call new class instances from within a class method itself (see the truediv function closely)