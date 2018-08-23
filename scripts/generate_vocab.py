#!/usr/bin/env python3

import sys
from collections import Counter
x = Counter(a.strip() for a in sys.stdin.readlines())
for y in x.keys():
    if y.lower() != y and y.lower() in x:
        x[y.lower()] += x[y]
        x[y] = 0
i = 0
for y in x.keys():
    if x[y] > 1:
        print(i, y)
        i += 1
