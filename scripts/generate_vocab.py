#!/usr/bin/env python3

import sys
from collections import Counter

MINIMUM_OCCURRENCES = 2

x = Counter(a.strip() for a in sys.stdin.readlines())

# if the lower-case version of a word exists, use that instead
for y in x.keys():
    if y.lower() != y and y.lower() in x:
        x[y.lower()] += x[y]
        x[y] = 0

# ouput only words that occur enough times. Give each output a unique ID
i = 0
for y in x.keys():
    if y != "" and x[y] >= MINIMUM_OCCURRENCES:
        print(i, y)
        i += 1
