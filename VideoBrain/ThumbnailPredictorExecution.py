# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:16:32 2019

@author: Alessandro
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import scene_thumbnail_processor as tp



# Predictor parameters to tune
parameters = {
              'process_color' : True, #if False process based on Blur
              'process_faces' : True, #if True process musical videos
              'MAX_FRAMES' : 50,
              'n_max_frame' : 2,
              'log' : False,
              'indexes' : (0,10), #range of source videos to be analized (es. (0,10) for the first 10 videos)
              'domain':'music'
                }





if __name__ == '__main__':
    sourcefile = 'toprocess.json'
    outputFolder = f'{os.getcwd()}/experiments/thumbnails/Musical/27/07/2020'
    
   # tp.execute(sourcefile, outputFolder, **parameters)
    tp.executeSingle(outputFolder,**parameters)
