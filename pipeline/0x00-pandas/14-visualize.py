#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df.rename(columns={'Timestamp':'Date'},
          inplace=True)
df.index = pd.to_datetime(df.pop('Date'),
                          unit='s')
df['Close'].fillna(method='ffill',
          inplace=True)
df['High'].fillna(df['Close'],
                  inplace=True)
df['Low'].fillna(df['Close'],
                 inplace=True)
df['Open'].fillna(df['Close'],
                  inplace=True)
df['Volume_(BTC)'].fillna(0,
                          inplace=True)
df['Volume_(Currency)'].fillna(0,
                               inplace=True)
df = df.loc['2017-01-01 00:00:00':].resample('1D').agg(func={'High': 'max',
             'Low':'min',
             'Open':'mean',
             'Close':'mean',
             'Volume_(BTC)':'sum',
             'Volume_(Currency)':'sum'})
df.plot()
plt.show()
