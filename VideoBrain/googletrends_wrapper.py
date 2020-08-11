# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:01:10 2020

@author: vegex
"""

import pytrends
from pytrends.request import TrendReq

pytrends = TrendReq(hl='en-US', tz=360)
pytrends.trending_searches(pn='united_states')

