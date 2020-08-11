# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:05:40 2020

@author: Alessandro
"""

import cv2
import time
import generic_thumbnail_processor as proc
import pafy
import numpy as np
import ntpath
import os
from yolo import YOLO
from six.moves.urllib.parse import urlparse


##########  Utilities ##########
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def is_url(path):
    try:
        result = urlparse(path)
        return result.scheme and result.netloc and result.path
    except:
        return False


def getVideoId(url):
    id = ''
    if (is_url(url)):
        list = url.split('?v=')
        id = list[len(list)-1]
    return id

   
    
##########  Tagger class ##########
class Tagger(object):
    
    yolo = YOLO()
    
    
    def __init__(self, **kwargs):
        self.log = kwargs.get('log', False)
        self.process_color = kwargs.get('process_color', True)
        self.max_frames = kwargs.get('max_frames', 50)
        self.outputFolder = kwargs.get('outputFolder', './')
        
        
    
    def getBestVideo(self, video):
        best = video.getbest(preftype="mp4")
        resolution = best.resolution.split('x')
        url = best.url
        return url, resolution
  
    
    
    def downloadYTVideo(self, url, outfile):    
        videoPafy = pafy.new(url)
        url, dimensions =self.getBestVideo(videoPafy)     
        frame_width = int(dimensions[0])
        frame_height = int(dimensions[1])    
        out = cv2.VideoWriter(outfile,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
        cap = cv2.VideoCapture(url)
        while(True):
          ret, frame = cap.read()    
          if ret == True:     
            # Write the frame into the file 
            out.write(frame) 
            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
              break 
          # Break the loop
          else:
            break 
        # When everything done, release the video capture and video write objects
        cap.release()
        out.release()
        # Closes all the frames
        cv2.destroyAllWindows() 
    
    
    
    def getHistSeries(self, video):
        if is_url(video):
            videoPafy = pafy.new(video)
            video = self.getBestVideo(videoPafy)[0]
        cam = cv2.VideoCapture(video)
        series = list()
        while cam.isOpened():
            ret, frame = cam.read()
            if not ret:
                break
            if self.process_color:
                series.append(proc.image_colorfulness(frame))
            else:
                series.append(proc.estimate_blur(frame)[0])        
            key = cv2.waitKey(1) & 0xFF
            # Exit
            if key == ord('q'):
                break
        cam.release()
        return series
    
    
    
    def extractFrames(self, video, frame_series, outputFolder):       
        if is_url(video):
            v_id = getVideoId(video)
            videoPafy = pafy.new(video)
            video = self.getBestVideo(videoPafy)[0]
        else:
            v_id = path_leaf('.'.join(video.split('.')[:-1])).split('.')[0]
        workdir = proc.createWorkDir(v_id, outputFolder)        
        cam = cv2.VideoCapture(video)
        if not cam.isOpened():
            raise IOError('ExtractFrames Can\'t open "Yolo2Model"')
        frame_num = 0
        while cam.isOpened():
            ret, frame = cam.read()
            if not ret:
                return workdir
            blur_prediction = 0
            if (frame_num in frame_series) == True: 
                blur_prediction = proc.estimate_blur(frame)[0]
                cv2.imwrite(f'{outputFolder}/{v_id}/localMaxFrame_{frame_num}_{blur_prediction}.jpg', frame)          
            # Draw additional info
            frame_info = f'Frame: {frame_num}, Score: {blur_prediction}'
            cv2.putText(frame, frame_info, (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            key = cv2.waitKey(1) & 0xFF
            # Exit
            if key == ord('q'):
                break
            # Take screenshot
            if key == ord('s'):
                cv2.imwrite(f'frame_{time.time()}.jpg', frame)
            frame_num += 1
        cam.release()
    
    
    
    def getBOWfromMetadata(self, metadata):
        #bagOfWords initialization
        bow = dict()
        for imageMetaData in metadata:
            prediction = imageMetaData['predictions']        
            for pred in prediction:
                if pred['class'] in bow:
                    bow[pred['class']] += 1
                else:
                    bow[pred['class']] = 1
        return {k: v for k, v in sorted(bow.items(), key=lambda item: item[1], reverse = True)}
    
    
    
    def getKeywordsFromMetadata(self, metadata):
        kw = list()
        for imageMetaData in metadata:
            prediction = imageMetaData['predictions']        
            for pred in prediction:
                if pred['class'] not in kw:
                    kw.append(pred['class'])
        return kw
    
    
    
    def extractTagsFromVideo(self, video, bow = False):
        series = self.getHistSeries(video)
        frame_series = proc.extractFramesFromSeries(self.max_frames, series, log = self.log)
        workdir = self.extractFrames(video, frame_series, self.outputFolder)
        metadata = proc.batchHistCompare(workdir, self.yolo, process_faces = False, corr = 0.5)
        if bow:
            return self.getBOWfromMetadata(metadata)
        else:
            return self.getKeywordsFromMetadata(metadata)
