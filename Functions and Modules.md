---
layout: page
title: Functions and Modules
permalink: /Power of Introspection/
nav_order: 5
---

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

# Built-in Functions

## The str dir and callable Functions

1\. str function

str coerces data into a string. Every datatype can be coerced into a string

```python
>>> horsemen = ['war', 'pestilence', 'famine']

>>> str(horsemen)
"['war', 'pestilence', 'famine', 'Powerbuilder']"

>>> str(None)
'None'

>>> str(odbchelper)
"<module 'odbchelper' from 'c:\\docbook\\dip\\py\\odbchelper.py'>"
```

2\. dir function

This function returns a list of all possible methods in a list/dict/module

```python
>>> li = []
>>> dir(li)
['append', 'count', 'extend', 'index', 'insert', 'pop', 'remove', 'reverse', 'sort']

>>> d = {}
>>> dir(d)
['clear', 'copy', 'get', 'has_key', 'items', 'keys', 'setdefault', 'update', 'values']

>>> import odbchelper
>>> dir(odbchelper)
['__builtins__', '__doc__', '__file__', '__name__', 'buildConnectionString']

```


3\. callable funciton

It takes any object as input and returs <True> if the object can be called or returns <False> if uncallable

Let's take for example two built-in functions of the string module (these functions are now deprecated though)

string.punctuation returns a string of possible characters to be used in a string. However, the function itself is not callable to be used elsewhere.

```python
>>> string.punctuation
'!"#$%&\'()*+,−./:;<=>?@[\\]^_`{|}~'

>>> callable(string.punctuation)
False
```

Now lets try a new built-in function <join>

This function as we know can be called to join any two strings. Therefore:

```python
>>> callable(string.join)
True

```

All the above Built-in functions are present in a module called <__builtin__>  


# Getting Object References with getattr

The getattr function takes an input as a string, if that string referes to any method, then it returns that method itself

Closely look and understand the following example code. Also, note how the <pop> method is referenced as a string "pop".
Also, see how calling a list function like pop without the parenthesis does a similar job

```python
>>> li = ["Larry", "Curly"]

>>> li.pop
<built−in method pop of list object at 010DF884>

>>> getattr(li, "pop")
<built−in method pop of list object at 010DF884>

>>> getattr(li, "append")("Moe")
>>> li
["Larry", "Curly", "Moe"]

>>> getattr({}, "clear")
<built−in method clear of dictionary object at 00F113D4>
```

In the last line we see that the getattr function inputs a blank dict, however it still works becuase python knows the datatype is dict and that we are also asking for a dict method "clear". 

Note if we don't know what clear will do, we can always say:

```python
d = {}
print(d.clear.__doc__)
```

The above code will output the doc string for clear

## getattr as a Dispatcher

Let's assume that we have a module called <statstout> which has three functions:

output_html, output_xml, output_text

Now let's have a main output program which takes statstout as an input.

Here we see the power of the getattr function of accepting strings and relating them to module functions. Also, with this getattr, <b> we may/may not define an attribute to statsout.

```python
import statsout
def output(data, format="text"):
output_function = getattr(statsout, "output_%s" % format, statsout.output_text)
return output_function(data)
```

The return output_function statement runs the particular function in statstout based on the argument (data),
which is in turn based on the argument given to def output(data, format="text")

Note that we give getattr a third argument, which works as the default value in case the second argument is unavailable

## input and map functions

```python
points = list()
    for i in range(4):
        a = list(map(float, input().split()))
        points
```

__split__ is a string functions which splits the lines in a string.

__map__ is an in-built function which uses two arguments: a function and an iterable(list or dict)

```python
def addition(n):
    return n*2

numbers = {1:'one',2 : 'two', 3: 'three'}

result = map(addition,numbers.values())

print(list(result))

>> ['oneone', 'twotwo', 'threethree']
```

## Typecasting and converting to list all while taking user input

```python
a = list(map(float, input('enter value').split()))

print(a)
```

