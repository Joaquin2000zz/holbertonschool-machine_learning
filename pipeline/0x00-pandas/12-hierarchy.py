#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

df1.index = df1.pop('Timestamp')
df2.index = df2.pop('Timestamp')
df = pd.concat([df2.loc['1417411980':'1417417981'], df1.loc['1417411980':'1417417981']],
               keys=['bitstamp', 'coinbase']).reorder_levels([1, 0], axis=0)
df.sort_index(inplace=True)

print(df)
