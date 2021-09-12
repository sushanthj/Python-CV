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
finally:
  pass
```

A try...except block can have an else clause, like an if statement. If no exception is raised during the try block, the else clause is executed afterwards. <b>finally</b> runs no matter what.

In this case, that means that the from EasyDialogs import AskPassword import worked, so you should bind getpass to the AskPassword function.

# Working with Files

## Opening Files

When we use the __open__ bulit-in function to open a file, it creates a file object (like instantiating a file, but we don't get class instance, we just get a file) and we can open a file in different modes as well:

```
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

```
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

```
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

```
>>> from fileinfo import MP3FileInfo
>>> MP3FileInfo.__module__
'fileinfo'

>>> sys.modules[MP3FileInfo.__module__]
<module 'fileinfo' from 'fileinfo.pyc'>
```

### The os.path module

Suppose we have a specific module to concatenate two strings, or we have a separate module to split a string. This module can be OS specific, and therefore we may not know the __exact name of the module__<br>

In situations like this, we can make use of the os.path :

```
>>> import os
>>> os.path.join("c:\\music\\ap\\", "mahadeva.mp3")
'c:\\music\\ap\\mahadeva.mp3'

>>> (filepath, filename) = os.path.split("c:\\music\\ap\\mahadeva.mp3")
>>> filepath
'c:\\music\\ap'

>>> os.listdir("c:\\music\\_singles\\")
['a_time_long_forgotten_con.mp3', 'hellraiser.mp3',
'kairo.mp3', 'long_way_home1.mp3', 'sidewinder.mp3',
'spinning.mp3']
```

The os.listdir takes the path of a directory and just lists out the contents

Similarly there exists os.path.isfile(pathname), which outputs 1 if it is a file, or 0 if not.

```
>>> [f for f in os.listdir(dirname)
      if os.path.isdir(os.path.join(dirname, f))]

['cygwin', 'docbook', 'Documents and Settings']

```

# Regex

## Mundane patterns and re.search()

Let's define a pattern which we want to use as a baseline for comparison:

__Eg. pattern = '^M?M?M?$'__

Each symbol in the string pattern has a particular meaning:

- __^__ = start looking for the pattern only if that pattern exists in the beginning of the string.<br> If this didn't exist, then pattern can be found at any position of the string and it will return true

- __M?__ = to optionally match at least a single __M__ character if avialable. <br> However, since our string pattern has 3 such cases of __M?__, we will be matching anywhere between 1 and 3 chars

- __$__ is the antagonistic counterpart to the __^__. The dollar sign means to say that match should happen right uptill the end of the string.

- When we couple both __^__ and __$__, it means that the pattern should match exactly between start and end of string (i.e entire string should get matched), with no chars before or after the __M__'s

```
>>> import re
>>> pattern = '^M?M?M?$'

>>> re.search(pattern, 'MMM')
<SRE_Match object at 0106AA38>

>>> re.search(pattern, 'MMMM')
(No output here because $ insists that there should be no char after the third M, however there is another)

```
<i> Note. Even two chars 'MM' would have been identified by our pattern as it would just ignore the third M as it's optional

<i> Interstingly, evne an empty string would have matched our pattern as 

## n to m form

```
>>> pattern = '^M{0,3}$'

>>> re.search(pattern, 'M')

<_sre.SRE_Match object at 0x008EEB48>
```
This pattern says: "Match the start of the string, then anywhere from zero to three M characters, then the end of the string." 

The 0 and 3 can be any numbers; if you want to match at least one but no more than three M characters, you could say M{1,3}.

## Verbose REGEX

```
pattern = """
^                   # beginning of string
M{0,4}              # thousands − 0 to 4 M's
(CM|CD|D?C{0,3})    # hundreds − 900 (CM), 400 (CD), 0−300 (0 to 3 C's), or 500−800 (D, followed by 0 to 3 C's)
(XC|XL|L?X{0,3})    # ones − 9 (IX), 4 (IV), 0−3 (0 to 3 I's), or 5−8 (V, followed by 0 to 3 I's)
(IX|IV|V?I{0,3})
$                   # end of string
"""

```

Note that we are using python's interpretation of ruman numerals in a text file as an example for the above operations:

In Roman numerals, there are seven characters that are repeated and combined in various ways to represent numbers.
- I = 1
- V = 5
- X = 10
- L = 50
- C = 100
- D = 500
- M = 1000

# Matching normal digits and re.compile

Say we have phone numbers like __800-555-1212__

We can use a modfied form of regex like:

```
>>> phonePattern = re.compile(r'^(\d{3})−(\d{3})−(\d{4})$')

>>> phonePattern.search('800−555−1212').groups()

('800', '555', '1212')
```

- Always read regular expressions from left to right. This one matches the beginning of the string, and then (\d{3}). What's \d{3}? Well, the {3} means "match exactly three numeric digits"; it's a variation on the {n,m} syntax you saw earlier. \d means "any numeric digit" (0 through 9). 

- Putting it in parentheses means "match exactly three numeric digits, and then remember them as a group that I can ask for later". 

- Then match a literal hyphen. 
- Then match another group of exactly three digits. 
- Then another literal hyphen. 
- Then another group of exactly four digits. 
- Then match the end of the string.

To get access to the groups that the regular expression parser remembered along the way, use the
groups() method on the object that the search function returns. It will return a tuple of however
many groups were defined in the regular expression. In this case, you defined three groups, one with
three digits, one with three digits, and one with four digits.

## Handling different seperators

```
>>> phonePattern = re.compile(r'^(\d{3})\D+(\d{3})\D+(\d{4})\D+(\d+)$')

>>> phonePattern.search('800 555 1212 1234').groups()

('800', '555', '1212', '1234')
```

\D matches any character except a numeric digit, and + means "1 or more".<br>
So \D+ matches one or more characters that are not digits.


## Handling numbers without seperators

```
>>> phonePattern = re.compile(r'^(\d{3})\D*(\d{3})\D*(\d{4})\D*(\d*)$')

>>> phonePattern.search('80055512121234').groups()

('800', '555', '1212', '1234')
```

__Instead of \D+ between the parts of the phone number, you now match on \D*. Remember that + means "1 or more"? Well, * means "zero or more".__

So now you should be able to parse phone numbers even when there is no separator character at all.

## Verbose number handlers

Taking the same example of phone numbers and designing a pattern matcher for that:

Eg. Phone number: work 1−(800) 555.1212 #1234

```
phonePattern = re.compile(r'''

(\d{3})   # area code is 3 digits (e.g. '800')

\D*       # optional separator is any number of non−digits
(\d{3})   # trunk is 3 digits (e.g. '555')
\D*       # optional separator
(\d{4})   # rest of number is 4 digits (e.g. '1212')
\D*       # optional separator
(\d*)     # extension is optional and can be any number of digits
$         # end of string
''', re.VERBOSE)

```
