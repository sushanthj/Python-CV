---
layout: page
title: Pandas Case Study
permalink: padas_case_study
nav_order: 15
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
{: .fs-9 }

[Reference](){: .btn .fs-5 .mb-4 .mb-md-0}

# Problem definition

Few ROS nodes were profiled to measure the transfer delays between ROS nodes in series. I wrote the outputs from each node into a text file; indexing each entry with the camera_id of that particular image

Now I had multiple text files, with entries like:
```
camID 9550 CD_in_time 1648216864996326208
camID 9551 CD_in_time 1648216865085628986
```

The above lines were for one node only (the input to one node). Now, to process such data from all nodes, I had to convert these rows into a **pandas dataframe**

# Solution Implementation

## Reading from a text file:

We will be reading from a folder and extracting all text files from there.

```python
from collections import defaultdict
from copy import copy, deepcopy
from operator import index
import pandas as pd
import csv
import os
import copy
import glob

from scipy.misc import electrocardiogram

my_dir_path = "/home/sush/TS/e2e_test/eval_sj"

collated_data = defaultdict(list)
file_count = 0
for file in sorted(glob.glob(os.path.join(my_dir_path, '*.txt'))):
    file_count += 1
    with open(file, "r") as file_open:
        for line in file_open:
            # add the contents of each line to a list
            elements = line.split("\n")[0].split('camID ')[1].split(" ")
            # save data according to camID (which is the element[0] in each line)
            collated_data[elements[0]].append(elements[1])
            collated_data[elements[0]].append((float(elements[2])*1000))
        
    # Note. If the data for each index (here camID) does not match the lenght of data in other datapoints, 
    # pandas cannot create a dataframe
    collated_data_clone = copy.deepcopy(collated_data)
    for cam_id, values in collated_data_clone.items():
        # check if any entry (whose key is camID) is not recoreded in each node
        if len(values) < (file_count * 2):
            # if data missing in any node: delete the data for that particular camID
            del collated_data[cam_id]

dframe = pd.DataFrame.from_dict(collated_data)
# the above created df will have camID as columns, we want it to be row-wise therefore transpose
dframe = dframe.transpose()
# sort the columns according to name

proc_time_column_no = 0
handover_column_no = 1
print("text file count is", file_count)
for i in range(int(file_count/2)):
    if i < file_count-1:
        proc_time_column_no += 4
        handover_column_no += 4
        dframe.insert(proc_time_column_no, 'proc_time_' + str(i), None)
        dframe.insert(handover_column_no, 'handover_' + str(i), None)
    else:
        proc_time_column_no += 4
        dframe.insert(proc_time_column_no, 'proc_time_' + str(i), None)

#print(dframe)

for i in range(int(file_count/2)):
    time_columns = [((i+1)*4)-1, ((i+1)*4)-3] 
    preproc_column_name = "proc_time_" + str(i)
    dframe[preproc_column_name] = dframe[time_columns[0]] - dframe[time_columns[1]]

#print(dframe)
dframe.to_csv(os.path.join(my_dir_path, 'evaluation.csv'))
```