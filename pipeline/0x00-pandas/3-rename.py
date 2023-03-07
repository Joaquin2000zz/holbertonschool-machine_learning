#!/usr/bin/env python3

import pandas as pd


from_file = __import__('2-from_file').from_file

df = from_file(filename='coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv',
               delimiter=',')
df.rename(columns={'Timestamp':
                   'Datetime'}, inplace=True)
df['Datetime'] = pd.to_datetime(
                                df.get('Datetime'),
                                unit='s')
old_df = df

df = df[['Datetime', 'Close']]

print(df.tail())
