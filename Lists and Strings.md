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

```
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

```
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

```
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

```
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

```
>>> range(7) = [1, 2, 3, 4, 5, 6]
(MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY) = range(7)

>>> MONDAY
0
```

6\. Looping in Lists

```
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

1. Formatting for I/O

```
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


2. Looping over Dictionary elemetns

```
>>> params = {"server":"mpilgrim", "database":"master", "uid":"sa", "pwd":"secret"}

>>> [k for k, v in params.items()]
['server', 'uid', 'database', 'pwd']

>>> ["%s=%s" % (k, v) for k, v in params.items()]
['server=mpilgrim', 'uid=sa', 'database=master', 'pwd=secret']
```

3. Joining String items in a dictionary with a secondary string

You might have thought I meant that string variables are objects. But no, look closely at this example and
you'll see that the string ";" itself is an object, and you are calling its join method.

The join method joins the elements of the list into a single string, with each element separated by a semi−colon. The
delimiter doesn't need to be a semi−colon; it doesn't even need to be a single character. It can be any string.

```
>>> params = {"server":"mpilgrim", "database":"master", "uid":"sa", "pwd":"secret"}

>>> ";".join(["%s=%s" % (k, v) for k, v in params.items()])
'server=mpilgrim;uid=sa;database=master;pwd=secret'
```

4. Splitting a string

Understand that split reverses join by splitting a string into a multi−element LIST. 

Note that the delimiter (";") is stripped out completely; it does not appear in any of the elements of the returned list.

```
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

```
>>> li = ["a", "mpilgrim", "foo", "b", "c", "b", "d", "d"]

>>> [elem for elem in li if len(elem) > 1]
['mpilgrim', 'foo']

>>> [elem for elem in li if li.count(elem) == 1]
['a', 'mpilgrim', 'foo', 'c']

```

count(list) is a list method to return the number of times a value occurs in a list. 

# Basic String and List Operations

## Finding Length of a String

```
name = "sush"
print(len(name))
```

## Extracting characters from a string into a list

```
str1 = [char for char in name]
```

## Looping over a list

'len' of list a = [a,b,c,d] gives output as 4. But there are only 3 indeces in our example list.

Therefore we use a small trick to utilize the integer output given by len(list)

```
thislist = ["apple", "banana", "cherry"]
for i in range(len(thislist)):
  print(thislist[i])
```

## Looping over the list of strings to access index and item values

In the code below note that index comes first and then item value when we do enumerate.

i.e. list(enumerate) => [(index1, item1), (index2, item2)]

```
for pos, letter in enumerate(str1):
        if letter in vowels:
            vowel_list.append(letter)
            vowel_index_list.append(pos)
```

Also remember that if we only say

```
list1 = ['B', 'F', 'F']
for i in list1:
  print(i)

>>B
>>F
>>F
```

From above we can see that 'i' represents the actual item in the list and not it's index

## Accessing String Line-by-Line

We use the <split()> method here but the arguement we pass is <(\n)> 

```
 for line in my_string.split('\n'):
        print line
````
The above code will return the first line of the string

However, we would need each line of a string seperately, Then we could use the code below:

```
txt = '''apple
       banana
       cherry
       orange'''

x = txt.split('\n')
```


# Small Note on Triangle loops

## Palindromes

Palindromes like 1, 121, 12321, 1234321 are actually squares of multiples of 11. See this simple code below:

```
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

```
for i in range(1,int(input())):
    print(int(i * 10**i / 9))

>>1
  22
  333
  4444

```
