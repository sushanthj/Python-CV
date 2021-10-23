---
layout: page
title: File Ops and Parsing
permalink: /File_Ops_and_Parsing/
nav_order: 9
---

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>


# Refer

refer: [Conv](https://diveintopython3.net/files.html)

# Built-in Functions:

## Open and close a file automatically

In the following example we will use the file.read() and file.seek() functions:

- file.seek(17) goes to the 17th byte of the file
- file.read(1) reads 1 character from the file at the current position of the pointer

```python
with open('examples/chinese.txt', encoding='utf-8') as a_file:
    a_file.seek(17)
    a_character = a_file.read(1)
    print(a_character)
```

## To print one line at a time

Note. {:>4} means that print line_number right justified within 4 spaces (see example below)

Observe the with block of code

```python
line_number = 0
with open('examples/favorite-people.txt', encoding='utf-8') as a_file:  ①
    for a_line in a_file:                                               ②
        line_number += 1
        print('{:>4} {}'.format(line_number, a_line.rstrip()))
```

