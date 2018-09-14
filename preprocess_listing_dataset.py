#!/usr/bin/python
# -*- coding: utf-8 -*-
""" 
This script reads the geojson file, extract features, transform geometry 
feature into shapely MultiPolygon objects and save the result as pickle file.

Author: Xian Lai
Date: Mar.20, 2018
"""

import ast

raw_listings_path = "data/listings_raw.csv"
processed_listings_path = "data/listings_preprocessed.pkl"
usecols = [
    'bathrooms', 'bedrooms', 'created', 'features', 'interest_level', 
    'latitude', 'longitude', 'price'
]

df = pd.read_csv(raw_listings_path, usecols=usecols)
df = df.dropna().reset_index(drop=True).copy()
df = df.rename({'created':'listing_time', 'longitude':'x', 'latitude':'y'}, axis=1)

# parse listing time
timedelta = pd.to_datetime(df['listing_time']) - pd.to_datetime("2016-01-01")
df['listing_time'] = (timedelta / np.timedelta64(1, 'h')).apply(lambda x: np.floor(x))

# numerize interest_level
dict_ = {'low':0, 'medium':1, 'high':2}
df['interest_level'] = df['interest_level'].apply(lambda x: dict_[x])

# concatenate feature strings and remove spaces
def concat_features(x):
    list_ = ast.literal_eval(x)
    res = ''
    for f in list_:
        res += f
    res = res.lower()
    res = res.replace(" ", "")
    return res

df['features'] = df['features'].apply(concat_features)

# standard scale continuous features
def standard_scale(col):
    global df
    df[col] = df[col].astype(float)
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

for col in ['bathrooms', 'bedrooms', 'listing_time', 'interest_level', 'price']:
    standard_scale(col)


# add centroid feature
df['centroid'] = df.apply(lambda row: (row['x'], row['y']), axis=1)

# save result as csv file
df.to_pickle(processed_listings_path)

print("Total number of listings: %d" %len(df))
print("\nPreprocessed listing dataset:")
print(df.head())








