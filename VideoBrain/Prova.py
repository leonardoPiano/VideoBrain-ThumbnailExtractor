# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:49:50 2020

@author: vegex
"""

import time
from tagger import Tagger
import os

url = 'https://www.youtube.com/watch?v=RUWKl1fR_9A'
outfile : 'outpy.avi'

parameters = {'outputFolder' : f'{os.getcwd()}/experiments/thumbnails',
              'log' : False,
              'max_frames' : 50,
              'process_faces' : True       
              }

tagger= Tagger(**parameters)


start_time = time.time()
kw = tagger.extractTagsFromVideo(url)
end_time = time.time()
delta_time = end_time - start_time
print(f'Execution completed in : {delta_time} seconds')
