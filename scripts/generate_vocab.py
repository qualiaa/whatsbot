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

# ouput only words that occur enough times.
del x[""]

x.subtract(list(x) * (MINIMUM_OCCURRENCES-1))
x = +x

# sort by count
x = sorted(list(x.items()),key=lambda a: a[1],reverse=True)

# give each output a unique ID
for i, w in enumerate((a[0] for a in x), start=1):
    print(i, w)
