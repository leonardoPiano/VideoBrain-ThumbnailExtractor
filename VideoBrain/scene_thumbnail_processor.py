# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:20:55 2020

@author: leopi
"""

from yolo import YOLO
from PIL import Image
from six.moves.urllib.parse import urlparse
import glob
import ntpath
import logging
import os
import cv2
import time
import logging.config
import pafy
import numpy as np
import pandas as pd
import json
import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
from scenedetect.detectors import ThresholdDetector
STATS_FILE_PATH = 'testvideo.stats.csv'


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import shutil

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger()


class Predictor:
    """ This class manage the lodeing of the models as a singleton """
    __instance = None
    #public fields
    yolo = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if Predictor.__instance == None:
            Predictor()
        else: logger.info("prediction models already loaded.")
        return Predictor.__instance 

    def __init__(self):
        """ Virtually private constructor. """
        if Predictor.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            Predictor.__instance = self
            logger.info("loading models")
            # parameters for loading data and images
            Predictor.yolo = YOLO()
            logger.info("Yolo loaded.")


def cleanDir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))             

def estimate_blur(image, threshold=100):
    
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    score = np.var(blur_map)
    return score, bool(score < threshold)


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


def getBestVideo(isBest,video, log = False):
    if log: print("-- Getting best video")
    streams = video.streams
    #isBest=True
    # resolution 640x360 format mp4
    logger.info(streams)
    for s in streams:
        logger.info(str(s.resolution)+ ' '+str(s.extension))
        if (s.resolution=='640x360' and s.extension=='mp4' and (not isBest)):
                logger.info(str(s.resolution)+ ' '+str(s.extension))
                return s.url
    return video.getbest(preftype="mp4").url


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def createFolder(id):
    newpath = os.getcwd()+'/outputs1/'+id 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath


def createWorkDir(video_id, folder):
    newpath = folder + '/' + video_id
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    else:
        cleanDir(newpath)
    return newpath
def predict_img(frame,yolo,domain):
       prediction=[]
       image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       image = Image.fromarray(image)  
       prediction=yolo.detect_img(image,domain)
      # print(prediction)
       
       return prediction

def get_sharpness(frame):
    
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gy, gx = np.gradient(grayFrame)
        gnorm = np.sqrt(gx**2 + gy**2)
        return np.average(gnorm)
    
def differentsFrames(video):
    treshold=15
    if(video is  None):
        return
        
    #video_manager = cv2.VideoCapture(video)
    video_manager = VideoManager([video])       
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)    
    scene_manager.add_detector(ContentDetector(treshold))
  
    base_timecode = video_manager.get_base_timecode()

    try:
    
         # Set downscale factor to improve processing speed.
        video_manager.set_downscale_factor()

        # Start video_manager.
        video_manager.start()
        
        scene_manager.detect_scenes(frame_source=video_manager)
        # Obtain list of detected scenes.
        scene_list = scene_manager.get_scene_list(base_timecode)     
                 
        frames=[]
        for i, scene in enumerate(scene_list):         
                      
            frames.append(scene[0].get_frames())
          
                                      
                                 
    finally:
        video_manager.release()
    
    return frames


def getOnlyNFramesFaces(best_frames,n):
    while len(best_frames)>n:
        
         predictions=best_frames[0]['predictions']
         #min_area=(predictions[0]['area'])
         max_area=0
         frame_id=0
         counter=0
         for frame in best_frames:
             predictions= frame['predictions']
             
             for prediction in predictions:
                 if(prediction['area']>max_area):
                     max_area=prediction['area']
                     frame_id=counter
             counter+=1
         del best_frames[frame_id]
    
    return best_frames
        
def getOnlyNFramesYolo(best_frames,n,log=False):
    
    while len(best_frames)>n:
        predictions=best_frames[0]['predictions']
        
        min=float(predictions[0]['score'])
        frame_id=0
        counter=0
        for frame in best_frames:
             predictions=frame['predictions']
             
             
             
             for prediction in predictions:              
                 
                 if(float(prediction['score'])<min):                     
                    frame_id=counter
                    min=prediction['score']
             counter+=1
                    
       
        del best_frames[frame_id]
    
    return best_frames


def getOnlyFrameSharp(best_frames,n,log=False):
    while len(best_frames)>n:

            min_sharp=best_frames[0]['sharpness']
           
            frame_id=0
            counter=0
            for frame in best_frames:
                sh=frame['sharpness']                
                if(sh<min_sharp):                  
                    frame_id=counter
                    min_sharp=sh
                counter+=1
            
            
            del best_frames[frame_id]
    
    return best_frames





        
def getFinalThumbnails(workdir,metadata,n,domain):
    
    bestFrames=[]
   
    max_score=0
    print( "FRAMES  ", len(metadata))
    if(len(metadata)>n):
        
    
        for frame in metadata:       
            
            predictions=frame['predictions']
            
            best=None       
            for prediction in predictions:
                                  
                if(float(prediction['score'])>max_score):                    
                    best=frame
                    
                    
                    
            if best is not None:
                bestFrames.append(best)
           
        
                            
        if(len(bestFrames)>n and domain !='music'):
            bestFrames=  getOnlyNFramesYolo(bestFrames,n)
        else :
            bestFrames=getOnlyNFramesFaces(bestFrames,n)
        
        for file in bestFrames:
            fnumb = file['fnumb']
            fname=file['fname']
            old=workdir+'/'+fname
            new_name = workdir+ '/'+'finalThumb '+str(fnumb)+'.jpg'
            os.rename(old, new_name)
       
        images = glob.glob(workdir+'/'+'*')
        
        for filename in images:
           
            if ('localMaxFrame' in filename)   :
                os.unlink(filename)


                
            
                    
     
            
          
 
    
    
    
def getImgseries(video_url,domain,yolo,outputFolder,v,n,log=False):
    v=''
    prediction=[]
    metadata=[]
    series = list()
    framesList=[]
    start_time=time.time()
    video = video_url
    incr = 0.1
    if log: print("video_url "+str(video_url))
    #it returns the id if is the url of a youtube video
    v_id = getVideoId(video)
  
    if log: print("v_id "+str(v_id))
    if is_url(video):
        if log: print("before pafy")
        videoPafy = pafy.new(video)
        if log: print("after pafy")
        video = getBestVideo(True,videoPafy, log = log)
    else:
        v_id = v

    workdir = createWorkDir(v_id, outputFolder)
       
    framesList=differentsFrames(video)

    print("Detected Frames : " ,len(framesList))
    cam = cv2.VideoCapture(video)
    if not cam.isOpened():
        raise IOError('Can\'t open Yolo2Model')
    frame_num = 0
   
    fps = 0
   
    try:
        
        while True:
            ret, frame = cam.read()
           
            if not ret:
                
                if log: print('Can\'t read video data. Potential end of stream') 
               
                return getFinalThumbnails(workdir,metadata,n,domain) 
            if frame_num in framesList:
                #sh= get_sharpness(frame) #estimate_blur(frame)[0]
                if(domain=='music'):
                    prediction=predictFaces(frame)
                else:
                    
                    prediction=predict_img(frame,yolo,domain)
                '''
                if(len(prediction)==0):
                  
                    if(sh>80):
                        fname='localMaxFrame_'+str(frame_num)+'.jpg' 
                        meta_info={}
                        meta_info['fname']=fname
                        meta_info['predictions']=prediction
                        meta_info['fnumb']=frame_num
                        meta_info['sharpness']=sh
                        metadata.append(meta_info)
                        cv2.imwrite(f'{outputFolder}/{v_id}/localMaxFrame_{frame_num}.jpg', frame)
                        '''
                        
                if len(prediction)>0:                    
                    meta_info={}
                    fname='localMaxFrame_'+str(frame_num)+'.jpg'                  
                    cv2.imwrite(f'{outputFolder}/{v_id}/localMaxFrame_{frame_num}.jpg', frame)
                    meta_info['fname']=fname
                    meta_info['predictions']=prediction
                    meta_info['fnumb']=frame_num
                  #  meta_info['sharpness']=sh
                    metadata.append(meta_info)
                if len(prediction)==0 and len(framesList)<=n:
                    meta_info={}
                    fname='localMaxFrame_'+str(frame_num)+'.jpg'                  
                    cv2.imwrite(f'{outputFolder}/{v_id}/localMaxFrame_{frame_num}.jpg', frame)
                    meta_info['fname']=fname
                    meta_info['predictions']=prediction
                    meta_info['fnumb']=frame_num
                  #  meta_info['sharpness']=sh
                    metadata.append(meta_info)
                    
                    
                    
                                              
            end_time = time.time()            
            fps = fps * 0.9 + 1/(end_time - start_time) * 0.1
            start_time = end_time
            time.sleep(0.000001)           
            key = cv2.waitKey(1) & 0xFF
      
            # Exit
            if key == ord('q'):
                break
            frame_num += 1
          
       
        
    finally:
        cam.release()


   
def processing(youtube,yolo_v3, outputFolder, **kwargs):
    #Loading parameters
    domain=kwargs.get('domain')
    n_max_frame = kwargs.get('n_max_frame', 5)
    log = kwargs.get('log', False)
    process_color = kwargs.get('process_color', True)
    process_faces = kwargs.get('process_faces', True)
    max_frames = kwargs.get('max_frames', 50)
    #Starting processing
    if log: print("Processing at: "+os.getcwd())
    yolo = yolo_v3
    yt = youtube
        
    getImgseries(yt,domain,yolo,outputFolder,'',n_max_frame,log=log)  
  
    print("-- Frame series ok")
 
    print("-- Extract frame ok")
   






def predictFaces(image, log = False):
    if log: print('predictFaces: ',image)
    # Load the cascade
    t = os.getcwd()
    face_cascade = cv2.CascadeClassifier(t + '/haarcascade_frontalface_default.xml')
    if log: print('face_cascade')
    img = image#cv2.imread(image)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    predictions = {
                    "predictions": []
                    }
    predictions=[]
    # add the faces
    for (x, y, w, h) in faces:
        prediction = {
                        "area": int(w*h),
                        "box": [
                                int(x),
                                int(y),
                                int(w),
                                int(h)
                                ],
                        "class": "face",
                        "score": 0.99
                     }
        predictions.append(prediction)

    return predictions



def executeSingle(outputFolder,**kwargs):
    yolo=YOLO()
    test='Atlantic Remaining Thumbnails.xlsx'
    test1='TreySongz remaining thumbnails.xlsx'
    test2='thumbnails_2020_07_27_DO_NOT_USE_WITH_OPTIMIZER.csv'
    toProcess= []
    indexes = kwargs.get('indexes', None)
    musicVideo= kwargs.get('process_faces')
    domain=kwargs.get('domain')
    if musicVideo:
        data=pd.read_csv(test2)
        toProcess=['https://www.youtube.com/watch?v='+l for l in data['itemCode']]
        
        
    else:
        
        with open( domain+'-'+'videos.txt')as f:
            toProcess=[l.strip() for l in f]
    
    if indexes is not None:
        toProcess = toProcess[indexes[0] : indexes[1]]
    
   
    for youtube in toProcess:
        try:
            print('Processing video '+youtube)
            start=time.time()
            processing(youtube, yolo, outputFolder, **kwargs)
            print('Processing video completed!!')
            print('Time: ',(time.time()-start))
        except:
            logging.exception('ERROR')
            print(youtube + " cannot be processed.")          
        
    