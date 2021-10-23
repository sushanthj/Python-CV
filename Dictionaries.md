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

