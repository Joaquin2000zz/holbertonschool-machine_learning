#!/usr/bin/env python3
"""
python script that creates a pd.DataFrame from a dictionary:

- The first column should be labeled First and have the values
  0.0, 0.5, 1.0, and 1.5
- The second column should be labeled Second and have the values
  one, two, three, four
- The rows should be labeled A, B, C, and D, respectively
- The pd.DataFrame should be saved into the variable df
"""
import inflect
import numpy as np
import pandas as pd


n = 4
p = inflect.engine()
df = pd.DataFrame({'First': np.arange(0, n / 2, .5),
                   'Second': [p.number_to_words(x) for x in range(1, n + 1)]},
                   index=[chr(65 + x) for x in range(n)])
