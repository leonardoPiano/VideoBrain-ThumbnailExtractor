# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:00:55 2019

@author: Alessandro
"""
import pandas
import collections


fileName = './report_2019_10_18.csv'

df = pandas.read_csv(fileName)

videoIDs = df['videocode']
channels = df['channelcode']
desc = df['description']

print(len(videoIDs.values))
print(len(set(videoIDs.values)))

print(len(desc.values))
print(len(set(desc.values)))


l = [item for item, count in collections.Counter(desc.values).items() if count > 1]