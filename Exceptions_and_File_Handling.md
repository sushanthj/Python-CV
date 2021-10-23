---
layout: page
title: Exceptions and File Handling
permalink: /Exceptions+File_Handlers/
nav_order: 7
---

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

# Intro

- Accessing a non−existent dictionary key will raise a KeyError exception.
- Searching a list for a non−existent value will raise a ValueError exception.
- Calling a non−existent method will raise an AttributeError exception.
- Referencing a non−existent variable will raise a NameError exception.
- Mixing datatypes without coercion will raise a TypeError exception.

# The 'else' part of try and except:

```python
try:
  from EasyDialogs import AskPassword
except ImportError:
  getpass = default_getpass
else:
  getpass = AskPassword
finally:
  pass
```

A try...except block can have an else clause, like an if statement. If no exception is raised during the try block, the else clause is executed afterwards. <b>finally</b> runs no matter what.

In this case, that means that the from EasyDialogs import AskPassword import worked, so you should bind getpass to the AskPassword function.

# Working with Files

## Opening Files

When we use the __open__ bulit-in function to open a file, it creates a file object (like instantiating a file, but we don't get class instance, we just get a file) and we can open a file in different modes as well:

```python
print open.__doc__  #displays all possible modes

>>> f = open("/music/_singles/kairo.mp3", "rb")
>>> f
>>> <open file '/music/_singles/kairo.mp3', mode 'rb' at 010E3988>
```

## Built-in File Methods

f.tell() = outputs current position of pointer in file <br>
f.seek( #bytes to consider, start from start/current/end pos) = moves pointer according 2 arguments provided <br>
f.read(128) = reads 128 bytes from file and returns data as a string <br>
f.close = closes a file <br>
f.closed = returns boolean depending upon whether file open or not <br>

f.write('some string') = adds the string to the file wherever the current pointer is

```python
>>> logfile = open('test.log', 'w')
>>> logfile.write('test succeeded')
>>> logfile.close()
>>> print file('test.log').read()
test succeeded
```
<i> Note. <b>file</b> is a synonym for <b>open</b>. This one−liner opens the file, reads its contents, and prints them.

### The sys Module

sys.modules is a dictionary containing all python modules which were imported since you started the IDE.<br>
As we know, each dictionary is a key value pair.<br>
So in this case, __key = module name, value = module object__

```python
>>> import sys
>>> print '\n'.join(sys.modules.keys())
win32api
os.path
os
exceptions
__main__
ntpath
nt
sys
__builtin__
site
signal
UserDict
```

Passing a module name as an argument to the sys.modules will give its location:

```python
from fileinfo import MP3FileInfo
MP3FileInfo.__module__
>>>'fileinfo'

sys.modules[MP3FileInfo.__module__]
>>><module 'fileinfo' from 'fileinfo.pyc'>
```

### The os.path module

Suppose we have a specific module to concatenate two strings, or we have a separate module to split a string. This module can be OS specific, and therefore we may not know the __exact name of the module__<br>

In situations like this, we can make use of the os.path :

```python
import os
os.path.join("c:\\music\\ap\\", "mahadeva.mp3")
>>>'c:\\music\\ap\\mahadeva.mp3'

(filepath, filename) = os.path.split("c:\\music\\ap\\mahadeva.mp3")
filepath
>>>'c:\\music\\ap'

os.listdir("c:\\music\\_singles\\")
>>>['a_time_long_forgotten_con.mp3', 'hellraiser.mp3',
'kairo.mp3', 'long_way_home1.mp3', 'sidewinder.mp3',
'spinning.mp3']
```

The os.listdir takes the path of a directory and just lists out the contents

Similarly there exists os.path.isfile(pathname), which outputs 1 if it is a file, or 0 if not.

```python
>>> [f for f in os.listdir(dirname)
      if os.path.isdir(os.path.join(dirname, f))]

['cygwin', 'docbook', 'Documents and Settings']

```