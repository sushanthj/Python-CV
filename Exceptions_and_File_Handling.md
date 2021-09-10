---
layout: page
title: Exceptions and File Handling
permalink: /Exceptions+File_Handlers/
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

# Intro

- Accessing a non−existent dictionary key will raise a KeyError exception.
- Searching a list for a non−existent value will raise a ValueError exception.
- Calling a non−existent method will raise an AttributeError exception.
- Referencing a non−existent variable will raise a NameError exception.
- Mixing datatypes without coercion will raise a TypeError exception.

# The 'else' part of try and except:

```
try:
from EasyDialogs import AskPassword
except ImportError:
getpass = default_getpass
else:
getpass = AskPassword
```

A try...except block can have an else clause, like an if statement. If no exception is raised during the try block, the else clause is executed afterwards. 

In this case, that means that the from EasyDialogs import AskPassword import worked, so you should bind getpass to the AskPassword function.

# Working with Files

## Opening Files

When we use the __open__ bulit-in function to open a file, it creates a file object (like instantiating a file, but we don't get class instance, we just get a file) and we can open a file in different modes as well:

```Python
print open.__doc__ displays all possible modes
```


```
>>> f = open("/music/_singles/kairo.mp3", "rb")
>>> f
<open file '/music/_singles/kairo.mp3', mode 'rb' at 010E3988>
```

