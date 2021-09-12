---
layout: page
title: Lists and Strings
permalink: /Lists and Strings/
nav_order: 3
---

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

# Lists and Strings

## Lists

1\. Slicing of Lists (common across numpy arrays as well)

```python
>>> li
['a', 'b', 'mpilgrim', 'z', 'example']
>>> li[:3]
['a', 'b', 'mpilgrim']
>>> li[3:]
['z', 'example']
>>> li[:]
['a', 'b', 'mpilgrim', 'z', 'example']
```
If the left slice index is 0, you can leave it out, and 0 is implied. So li[:3] is the same as li[0:3]

Similarly, if the right slice index is the length of the list, you can leave it out. So li[3:] is the same as
li[3:5], because this list has five elements.

Important Points to note
- list has indeces starting from 0,1,2...
- However len(string) and len(list) will give numbers counted as 1,2,3,4
- li[:2] outputs li[0] and li[1]
- li[1:4] outputs li[1], li[2] and li[3]


2\. Appending of Lists

```python
>>> li
['a', 'b', 'mpilgrim', 'z', 'example']
>>> li.append("new")
>>> li
['a', 'b', 'mpilgrim', 'z', 'example', 'new']
>>> li.insert(2, "new")
>>> li
['a', 'b', 'new', 'mpilgrim', 'z', 'example', 'new']
>>> li.extend(["two", "elements"])
>>> li
['a', 'b', 'new', 'mpilgrim', 'z', 'example', 'new', 'two', 'elements']
```
Append adds a single element to the end of the list.

Insert inserts a single element into a list at a specified location and the original placeholder at that index gets bumped to the next index
Extend concatenates lists. 

3\. Removing Items

```python
>>> li
['a', 'b', 'new', 'mpilgrim', 'z', 'example', 'new', 'two', 'elements']
>>> li.remove("z")
>>> li
['a', 'b', 'new', 'mpilgrim', 'example', 'new', 'two', 'elements']
>>> li.remove("new")
>>> li.pop()
'elements'
>>> li
['a', 'b', 'mpilgrim', 'example', 'new', 'two']
```

4\. List Operators

```python
>>> li = ['a', 'b', 'mpilgrim']
>>> li = li + ['example', 'new']
>>> li
['a', 'b', 'mpilgrim', 'example', 'new']
>>> li += ['two']
>>> li
['a', 'b', 'mpilgrim', 'example', 'new', 'two']
>>> li = [1, 2] * 3
>>> li
[1, 2, 1, 2, 1, 2]
```

5\. Assignment to Variables

```python
>>> range(7) = [1, 2, 3, 4, 5, 6]
(MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY) = range(7)

>>> MONDAY
0
```

6\. Looping in Lists

```python
>>> li = [1, 9, 8, 4]
>>> [elem*2 for elem in li]
[2, 18, 16, 8]

>>> li
[1, 9, 8, 4]
```
As we can see from the above, looping did not change the actual variables in the list, python uses a seperate list in memory to loop

To make sense of this, look at it from right to left. li is the list you're mapping. Python loops through li one
element at a time, temporarily assigning the value of each element to the variable elem. Python then applies
the function elem*2 and appends that result to the returned list.

## Strings

1\. Formatting for I/O

```python
uid = "sa"
>>> pwd = "secret"
>>> print pwd + " is not a good password for " + uid
secret is not a good password for sa
print "%s is not a good password for %s" % (pwd, uid)
secret is not a good password for sa

>>> userCount = 6
>>> print "Users connected: %d" % (userCount, )
Users connected: 6
>>> print "Users connected: " + userCount
Traceback (innermost last):
File "<interactive input>", line 1, in ?
TypeError: cannot concatenate 'str' and 'int' objects
```
Note here that (userCount,) is a tuple with one element. If we ask for only (userCount) python would throw an error


2\. Looping over Dictionary elemetns

```python
>>> params = {"server":"mpilgrim", "database":"master", "uid":"sa", "pwd":"secret"}

>>> [k for k, v in params.items()]
['server', 'uid', 'database', 'pwd']

>>> ["%s=%s" % (k, v) for k, v in params.items()]
['server=mpilgrim', 'uid=sa', 'database=master', 'pwd=secret']
```

3\. Joining String items in a dictionary with a secondary string

You might have thought I meant that string variables are objects. But no, look closely at this example and
you'll see that the string ";" itself is an object, and you are calling its join method.

The join method joins the elements of the list into a single string, with each element separated by a semi−colon. The
delimiter doesn't need to be a semi−colon; it doesn't even need to be a single character. It can be any string.

```python
>>> params = {"server":"mpilgrim", "database":"master", "uid":"sa", "pwd":"secret"}

>>> ";".join(["%s=%s" % (k, v) for k, v in params.items()])
'server=mpilgrim;uid=sa;database=master;pwd=secret'
```

4\. Splitting a string

Understand that split reverses join by splitting a string into a multi−element LIST. 

Note that the delimiter (";") is stripped out completely; it does not appear in any of the elements of the returned list.

```python
>>> s
'server=mpilgrim;uid=sa;database=master;pwd=secret'

>>> s.split(";")
['server=mpilgrim', 'uid=sa', 'database=master', 'pwd=secret']

>>> s.split(";", 1)
['server=mpilgrim', 'uid=sa;database=master;pwd=secret']
```

Note that split can also take a secondary argument which controls the number of splits done on the string. 

# List Filetering

In the below examples focus on the process of how we: 

Save -> each element in a list (by looping) -> if it satisfies some logic

```python
>>> li = ["a", "mpilgrim", "foo", "b", "c", "b", "d", "d"]

>>> [elem for elem in li if len(elem) > 1]
['mpilgrim', 'foo']

>>> [elem for elem in li if li.count(elem) == 1]
['a', 'mpilgrim', 'foo', 'c']

```

count(list) is a list method to return the number of times a value occurs in a list. 

# Basic String and List Operations

## Finding Length of a String

```python
name = "sush"
print(len(name))
```

## Extracting characters from a string into a list

```python
str1 = [char for char in name]
```

## Substituting specific words of a string

there exists a built-in function called re.sub which can do the following:

```python
import re
>>> s = '100 NORTH MAIN ROAD'

>>> re.sub('ROAD$', 'RD.', s)
'100 NORTH BROAD RD.'
```

The module __re__ means __'regular expression'__
Using the re.sub function, you search the string s for the regular expression 'ROAD$' and replace it with 'RD.'. <br> This matches the ROAD at the end of the string s, but does not match the ROAD that's part of the word BROAD, because that's in the middle of s.

However, a better way to do it would be to specify that ROAD should be a separate word on it's own.<br> We can do this by adding a clause r\bROAD$ which means that:<br>
Only if the raw string (r\) has a word boundary around the word 'ROAD', only then sub:

```python
'100 BROAD ROAD APT. 3'

>>> re.sub(r'\bROAD\b', 'RD.', s)

'100 BROAD RD. APT 3'
```

## Looping over a list

'len' of list a = [a,b,c,d] gives output as 4. But there are only 3 indeces in our example list.

Therefore we use a small trick to utilize the integer output given by len(list)

```python
thislist = ["apple", "banana", "cherry"]
for i in range(len(thislist)):
  print(thislist[i])
```

## Looping over the list of strings to access index and item values

In the code below note that index comes first and then item value when we do enumerate.

i.e. list(enumerate) => [(index1, item1), (index2, item2)]

```python
for pos, letter in enumerate(str1):
        if letter in vowels:
            vowel_list.append(letter)
            vowel_index_list.append(pos)
```

Also remember that if we only say

```python
list1 = ['B', 'F', 'F']
for i in list1:
  print(i)

>>B
>>F
>>F
```

From above we can see that 'i' represents the actual item in the list and not it's index

## Accessing String Line-by-Line

We use the `<split>` method here but the arguement we pass is <(\n)> 

```python
 for line in my_string.split('\n'):
        print line
```
The above code will return the first line of the string

However, we would need each line of a string seperately, Then we could use the code below:

```python
txt = '''apple
       banana
       cherry
       orange'''

x = txt.split('\n')
```

# Small Note on Triangle loops

## Palindromes

Palindromes like 1, 121, 12321, 1234321 are actually squares of multiples of 11. See this simple code below:

```python
for i in range(1,int(input())+1):
    print (((10**i - 1)//9)**2)

>>1
>>121
>>12321
>>.......
```

The <**> operator serves as an exponential and the <//> operator is used for floor division. i.e. 9//2 = 4

## Recurring numbers

Remember that any number divided by 9 (unless one of the number's factor is 9), throws recurrence

i.e 10/9 = 1.1111111. Let's make use of this in code to get the following:

```python
for i in range(1,int(input())):
    print(int(i * 10**i / 9))

>>1
  22
  333
  4444

```

## print(*text) function

The print of * for a text is equal as printing print(text[0], text[1], ..., text[n]) and this is printing each part with a space between. you can do:

```python
text = 'PYTHON'
for index in range(len(text))
    print("".join(list(text)[:index + 1]))
```

or you can just use the * operator as:

```python
text = 'PYTHON'
for index in range(len(text))
    print(*text[:index + 1], sep='')
```

