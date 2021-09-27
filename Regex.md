---
layout: page
title: RegEx
permalink: /Regex/
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

# Before you Begin

The regex functions mostly work with string inputs and basically do pattern matching. each expressions in a regex pattern is meant to match a specific part of a string

Here are the most important ones you'll be learning:

- \d = matches digits from 0-9
- \D = matches non-digits
- \s = matches whitespace
- \w = matches word characters

- ^ = matches beginning of line
- $ = matches end of line
- [..] = matches whatever chars we specify in the brackets
- [^..] = matches all characters other than those mentioned in brackets
- (\d{n}) = matches exactly n number of digits 

Also, we can do the following to match letters and numbers only:
```python
pattern = r"([a-zA-Z0-9])\1+"

m = re.search(pattern,input())

if m:
    print(m.group(1))
else:
    print(-1)
```
In the above program we are finding the first occurence of an alphanumeric in an input string


# Regex

In the code above, we saw that some in-built functions of the re class were used. Let's list them out with basic descriptions:

- re.search() = returns an object if a particular match is made for a pattern, we can then group this into a list using the re.group() function
- re.split() = similar to split function of string, but returns a list in this case
- re.sub() = replaces one or more matches with a string
- re.findall() = returns a list containing all matches
- re.group() = groups found objects (like the return type of re.search) into a list
- re.complie() = see example below (it also returns an object on which we will have to do another .search to get callable objects)
- re.match() = takes two arguments, i.e. the pattern to match and the string and returns a callable object which we can then group into a list if required

## Mundane patterns and re.search()

Let's define a pattern which we want to use as a baseline for comparison:

__Eg. pattern = '^M?M?M?$'__

Each symbol in the string pattern has a particular meaning:

- __^__ = start looking for the pattern only if that pattern exists in the beginning of the string.<br> If this didn't exist, then pattern can be found at any position of the string and it will return true

- __M?__ = to optionally match at least a single __M__ character if avialable. <br> However, since our string pattern has 3 such cases of __M?__, we will be matching anywhere between 1 and 3 chars

- __$__ is the antagonistic counterpart to the __^__. The dollar sign means to say that match should happen right uptill the end of the string.

- When we couple both __^__ and __$__, it means that the pattern should match exactly between start and end of string (i.e entire string should get matched), with no chars before or after the __M__'s

```python
>>> import re
>>> pattern = '^M?M?M?$'

>>> re.search(pattern, 'MMM')
<SRE_Match object at 0106AA38>

>>> re.search(pattern, 'MMMM')
#(No output here because $ insists that there should be no char after the third M, however there is another)

```
<i> Note. Even two chars 'MM' would have been identified by our pattern as it would just ignore the third M as it's optional

<i> Interstingly, evne an empty string would have matched our pattern as 

## n to m form

```python
>>> pattern = '^M{0,3}$'

>>> re.search(pattern, 'M')

<_sre.SRE_Match object at 0x008EEB48>
```
This pattern says: "Match the start of the string, then anywhere from zero to three M characters, then the end of the string." 

The 0 and 3 can be any numbers; if you want to match at least one but no more than three M characters, you could say M{1,3}.

## Verbose REGEX

```python
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

# Groups

## groups()

A groups() expression returns a tuple containing all subgroups of the match

```python
import re
m = re.match(r'(\w+)@(\w+)\.(\w+)','username@hackerrank.com')
m.groups()
>>> ('username', 'hackerrank', 'com')
```

## groupdict()

A groupdict() expression returns a dictionary containing all named subgroups of the match.

```python

import re
m = re.match(r'(?<user<\w+)@(>P<website>\w+)\.(>P<extension>\w+)' , 'myname@gmail.com')
m.groupdict()

>>> {'user': 'myname', 'website' : 'gmail', 'extension': 'com'}
```

In the above code we can notice that (\w+) would mean just a pattern of words, but we can specify a particular key value using the <string_key> operator

## findall()

The expression re.findall() returns non-ovealapping matches of patterns in a string -> as a list of strings

```python
import re
re.findall(r'\w', 'http://hackerrank.com/')
>>>['h', 't', 't', 'p', 'w', 'w', 'w', 'h', 'a', 'c', 'k', 'e', 'r', 'r', 'a', 'n', 'k', 'c', 'o', 'm']
```

However, we can replace `\w` with `\w+` to extract only words from the string

## finditer()

This expression returns a callable iterable object (iterator) over non-overlapping matches in string

```python
import re
re.finditer(r'\w', 'http://www.hackerrank.com/')
<callable-iterator object at 0x0266C790>

map(lambda x: x.group(),re.finditer(r'\w','http://www.hackerrank.com/'))

>>> ['h', 't', 't', 'p', 'w', 'w', 'w', 'h', 'a', 'c', 'k', 'e', 'r', 'r', 'a', 'n', 'k', 'c', 'o', 'm']
```

Remember that the map function uses 2 arguments:
- a function which it will iterate upon
- a list or dictionary of items as input

Therefore, we get the output as a list of letters in string

# Matching normal digits and re.compile

Say we have phone numbers like __800-555-1212__

We can use a modfied form of regex like:

```python
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

```python
>>> phonePattern = re.compile(r'^(\d{3})\D+(\d{3})\D+(\d{4})\D+(\d+)$')

>>> phonePattern.search('800 555 1212 1234').groups()

('800', '555', '1212', '1234')
```

\D matches any character except a numeric digit, and + means "1 or more".<br>
So \D+ matches one or more characters that are not digits.


## Handling numbers without seperators

```python
>>> phonePattern = re.compile(r'^(\d{3})\D*(\d{3})\D*(\d{4})\D*(\d*)$')

>>> phonePattern.search('80055512121234').groups()

('800', '555', '1212', '1234')
```

__Instead of \D+ between the parts of the phone number, you now match on \D*. Remember that + means "1 or more"? Well, * means "zero or more".__

So now you should be able to parse phone numbers even when there is no separator character at all.

## Verbose number handlers

Taking the same example of phone numbers and designing a pattern matcher for that:

Eg. Phone number: work 1−(800) 555.1212 #1234

```python
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
## Example: find vowels which are 2 chars in length and surrounded by consonants

```python
import re
v = "aeiou"
c = "qwrtypsdfghjklzxcvbnm"
print(*re.findall("(?=[%s]([%s]{2,})[%s])"%(c,v,c),input(), re.I) or [-1], sep="\n")
```

Note that the print(*..) part is just a cool way of saying:

```python
import re
v = "aeiou"
c = "qwrtypsdfghjklzxcvbnm"
#print(*re.findall("(?=[%s]([%s]{2,})[%s])"%(c,v,c),input(), re.I) or [-1], sep="\n")

pattern = []
pattern = re.findall("(?=[%s]([%s]{2,})[%s])"%(c,v,c),input(), re.I)

for i in range(len(pattern)):
    print(pattern[i])
```

## Example: Extracting only words from a string and index all words in a dictionary

```python
def get_words(inp_string):

  string_to_split = str(inp_string)
  pattern = '[A-Za-z]+'
  split_list = re.findall(pattern, string_to_split)
  return split_list

def create_dictionary(messages):

  word_dictionary = {}
    word_count = 0
    for i in range(messages.shape[0]):
        temp_list = get_words(messages[i])
        for j in range(len(temp_list)):
            if temp_list[j] not in word_dictionary.values():
                word_dictionary[word_count] = temp_list[j]
                word_count += 1

    return word_dictionary
```