---
layout: page
title: Dictionaries
permalink: /Dictionaries/
nav_order: 4
---

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

# Some interesting examples on dictionaries

## Dict update and delete

To update a dictionary and also delete some items in it, we may get a few errors if we do the following:

```python
for a,b in word_dictionary_init.items():
        if b >= 5:
            word_dictionary_init[a] = word_count
            word_count += 1
        else:
            del word_dictionary_init[a]
```

Here values of a and b are messed up because we keep deleting the items. Therefore a better way to do in THIS CASE is as follows:

```python
index = 0
    for word_key in list(word_dict.keys()):
        if word_dict[word_key] >= 5:
            word_dict[word_key] = index
            index += 1
        else:
            del word_dict[word_key]
    return word_dict
```
# Ordered Dictionaries

In case we want a dictionary to also follow structure like a list, we use these dictionaries

However, these dictionaries take up more space than normal dicts due to their doubly-linked lists backend. The process to create such a dict is shown below:

```python
from collections import OrderedDict

self.depth_dictionary = OrderedDict()

# access the dictonary like any other dictionary:
self.depth_dictionary['favourite_cake'] = 'pineapple'

# however, to remove the first item in such a dict, we have simple functions like below:
if len(self.depth_dictionary) > 10:
    self.depth_dictionary.popitem(last=False)

# in the above line, if last=True, then it will delete the last item of the dict
```
