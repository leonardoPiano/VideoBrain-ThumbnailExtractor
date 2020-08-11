# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:49:07 2020

@author: leopi
"""

import argparse
from yolo import YOLO
from filters import smooth
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
import json
import csv
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim

from scipy import signal
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


def estimate_blur(image, threshold=100):
    
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    score = np.var(blur_map)
    return score, bool(score < threshold)


def predict_img(img,yolo,domain,compair,corr,counter):
    prediction=[]   
    corr=0.90   
 
    if(compair is  None):
       dist=0

    else:
       
       dist=compareHist2(img,compair)  
 
       
    if(dist<=corr):
       #print(dist)
       image = Image.fromarray(img)  
       prediction=yolo.detect_img(image,domain)        
         
       
    if(len(prediction)>0):       
       compair=img
    
 
    return prediction,compair
    
def predict_img_old(img,yolo,domain,target,corr):
    prediction=[]
   
  
    dist=0
   
  
    if len(target)>0:       
       compair=target[len(target)-1] 
       dist=compareHist2(img,compair)
    
     
    if(dist<corr):          
       image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
       image = Image.fromarray(image)  
       prediction=yolo.detect_img(image,domain)
            
 
    return prediction
     
     
 

    
        
    
    
    
    
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


def compareHist(file1,file2,stat=cv2.HISTCMP_CORREL):
    target_im = cv2.imread(file1)
    target_hist = cv2.calcHist([target_im], [0], None, [256], [0, 256])
    comparing_im = cv2.imread(file2)
    comparing_hist = cv2.calcHist([comparing_im], [0], None, [256], [0, 256])
    diff = cv2.compareHist(target_hist, comparing_hist, stat)
    #print(diff)
    return diff 
def compareHist2(file1,file2,stat=cv2.HISTCMP_CORREL):
    target_im = file1
    target_hist = cv2.calcHist([target_im], [0], None, [256], [0, 256])
    comparing_im = file2
    comparing_hist = cv2.calcHist([comparing_im], [0], None, [256], [0, 256])
    diff = cv2.compareHist(target_hist, comparing_hist, stat)
    ##print(diff)
    return diff 



def batchHistCompare(dir,predictions,domain, process_faces = True, corr=0.99, log = False):
    #it includes image detection
    #yolo = YOLO()
    #param video 
    #param blur threshold
    #param frame correlation
    f_dist = corr
    metadata = []
   
    if dir: videos = [dir]
    for video in videos:
        try:
            logger.info(video)
            video_id = path_leaf(video)
            files = glob.glob(video+'/*')
            files.sort(key=os.path.getmtime)
            count = 0
            scene_counter = 0
            temp=0
            target = files[0]
            for file in files:
                try:
                    fname = path_leaf(file)
                    if log: print(file)
                    dist = compareHist(target,file)
                    target = file
                    isChanged = " no scene change."
                    if (dist<f_dist): 
                        isChanged = " THIS IS A NEW SCENE."
                        scene_counter = scene_counter +1
                       
                    if log: print(f'Image {count}, {path_leaf(file)} compare {dist}. {isChanged}')
                    metainfo = {}
                    metainfo['file'] = file
                    metainfo['frame'] = fname.split('_')[1]
                    metainfo['blur'] = fname.split('_')[2].split('.')[0]
                    metainfo['scene'] = scene_counter
                    metainfo['corr'] = dist
                    try:
                       
                      
                        if log: print('Image.opened: '+file)
                       
                    except:
                        print('File Open Error! Try again!')                                   
                                                                
                    metainfo['predictions']=predictions[count]
                    metadata.append(metainfo)
                    count = count + 1
                except:
                    logger.error(f'video id {video_id} not processed image {file}.', exc_info=True)
        except:
            logger.error(f'video id {video_id} not processed url {video}.')
    return metadata
    #this function draw the boxes around the detected objects

def getImgseries(video_url,domain,yolo,v='',log=False):
  
    compair=None
    metadata=[]
    series = list()
    target=[]
    video = video_url
    incr = 0.1
    if log: print("video_url "+str(video_url))
    #it returns the id if is the url of a youtube video
    v_id = getVideoId(video)
  
    if log: print("v_id "+str(v_id))
    if is_url(video):
        if log: print("before pafy")
        videoPafy = pafy.new(video)  
        #video = videoPafy.getbest(preftype="mp4").url
        if log: print("after pafy")
        video = getBestVideo(True,videoPafy, log = log)
    else:
        v_id = v

    cam = cv2.VideoCapture(video)
    if not cam.isOpened():
        raise IOError('Can\'t open Yolo2Model')
    frame_num = 0
    start_time = time.time()
    fps = 0
    count=0
    try:
        while True:
            ret, frame = cam.read()
            
            if not ret:
                if log: print('Can\'t read video data. Potential end of stream')            
                return series,metadata
            prediction=predict_img_old(frame,yolo,domain,target, 0.99)
           
            
            if len(target)<=2:
                target.append(frame)
            else: 
                target=[]
                target.append(frame)
           
            if len(prediction)>0:
                
                series.append(frame_num)
                metadata.append(prediction)
               
                                              
            end_time = time.time()            
            fps = fps * 0.9 + 1/(end_time - start_time) * 0.1
            start_time = end_time
            time.sleep(0.000001)           
            key = cv2.waitKey(1) & 0xFF
      
            # Exit
            if key == ord('q'):
                break
            frame_num += 1
        return series,metadata
        
    finally:
        cam.release()
    



def extractFrames(video_url,frame_series, outputFolder, v='', log = False):
    
    if log: print("extractFrames ok")
    video = video_url
    #it returns the id if is the url of a youtube video
    v_id = getVideoId(video)
    if log: print("extractFrames video: "+video)
    if is_url(video):
        videoPafy = pafy.new(video)
        #video = videoPafy.getbest(preftype="mp4").url
        video = getBestVideo(True,videoPafy, log = log)
    else:
        v_id = v    
    workdir = createWorkDir(v_id, outputFolder)        
    cam = cv2.VideoCapture(video)
    if not cam.isOpened():
        raise IOError('ExtractFrames Can\'t open "Yolo2Model"')
    frame_num = 0
    start_time = time.time()
    fps = 0

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                if log: print('ExtractFrames Can\'t read video data. Potential end of stream')
                return workdir
            blur_prediction = 0
            if (frame_num in frame_series):
                blur_prediction = estimate_blur(frame)[0]
               
                cv2.imwrite(f'{outputFolder}/{v_id}/localMaxFrame_{frame_num}_{blur_prediction}.jpg', frame)
               
            end_time = time.time()            
            fps = fps * 0.9 + 1/(end_time - start_time) * 0.1
            start_time = end_time
            time.sleep(0.000001)            
            # Draw additional info
            frame_info = f'Frame: {frame_num}, FPS: {fps:.2f}, Score: {blur_prediction}'
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
    finally:
        cam.release()
   
    
def generateThumb(workDir,exactFrames=5,log=False):
    workDir=workDir+'/'
    with open(workDir+'metadata.json') as f:
        metadata = json.load(f)
    lastScene = metadata[len(metadata)-1]
    to = int(lastScene['scene'])
    scenes = []
    for i in range(to+1):
        scenes.append([])
    for data in metadata:
        scenes[data['scene']].append(data)
    best_frames = []
    
    for i in range(to+1):
        frames=scenes[i]
        best_frame = None
        max_score = 0.5
        if (len(scenes[i])>0):
            best_frame = scenes[i][0]
        
            
        for frame in frames:
                
                if (len(frame['predictions'])>0):
                            max_area = 0
                            predictions = frame['predictions']                           
                            for prediction in predictions:
                                if (float(prediction['score'])>max_score):
                                    if(int(prediction['area'])>max_area):
                                        max_area = int(prediction['area'])
                                        best_frame = frame
        if best_frame is not None: 
                best_frames.append(best_frame)
       
    best_frames = getOnlyNFramesYolo(best_frames,exactFrames, log = log)
    with open(workDir+'/filtered_metadata.json', 'w') as f:
        json.dump(best_frames, f, indent=4, separators=(',', ': '), sort_keys=True)
    for file in best_frames:
        fname = path_leaf(file['file'])
        new_name = workDir+'selected_'+fname
        os.rename(file['file'], new_name)
    images = glob.glob(workDir+'*')
    for filename in images:
        if ('\localMaxFrame' in filename) or filename.endswith('.json')  :
            os.unlink(filename)
    return best_frames
            
            
    
    
def getColorResults(workDir, average_cut, exactFrames=5, log = False):
    workDir=workDir+'/'
    with open(workDir+'metadata.json') as f:
        metadata = json.load(f)
    lastScene = metadata[len(metadata)-1]
    to = int(lastScene['scene'])
    scenes = []
    for i in range(to+1):
        scenes.append([])
    for data in metadata:
        scenes[data['scene']].append(data)
    if log: 
        print("***************************")
        print(scenes)
        
    #best frame for scene max object area if blur > 1000
    best_frames = []
    for i in range(to+1):
        if log: print("*********** or i in range(to  ****************")
        frames = scenes[i]
        max_colorfullness = average_cut
        ref_colorfullness = average_cut
        max_score = 0.5
        best_frame = None
        if (len(scenes[i])>0):
            best_frame = scenes[i][0]
        if log: print("*************BEST FRAME**************")
        for frame in frames:
            if (int(frame['blur'])>max_colorfullness):
                max_colorfullness = int(frame['blur'])
                best_frame = frame
                if log: 
                    print("*************BEST FRAME**************")
                    print(best_frame)
        for frame in frames:
            if (int(frame['blur'])>ref_colorfullness):
                #seach for max object score 
                if (len(frame['predictions'])>0):
                    max_area = 0
                    predictions = frame['predictions']
                   
                    for prediction in predictions:
                        if (float(prediction['score'])>max_score):
                            if(int(prediction['area'])>max_area):
                                max_area = int(prediction['area'])
                                best_frame = frame
        if log: 
            print("************** Best frame i="+str(i))
            print(best_frame)
        if best_frame is not None: 
            best_frames.append(best_frame)
    if log: 
        print("*************************************")
        print(best_frames)
    best_frames = getOnlyNFrames(best_frames,exactFrames, log = log)
    with open(workDir+'/filtered_metadata.json', 'w') as f:
        json.dump(best_frames, f, indent=4, separators=(',', ': '), sort_keys=True)
    for file in best_frames:
        fname = path_leaf(file['file'])
        new_name = workDir+'selected_'+fname
        os.rename(file['file'], new_name)
    images = glob.glob(workDir+'*')
    for filename in images:
        if ('\localMaxFrame' in filename) or filename.endswith('.json')  :
            os.unlink(filename)
    return best_frames


def getOnlyNFramesYolo(best_frames,n=5,log=False):
    
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
        del best_frames[frame_id]
    return best_frames
def getOnlyNFrames(best_frames,n=5, log = False):
    if log: 
        print("*************************************")
        print(best_frames)
    while len(best_frames)>n:
        #remove min
        min = int(best_frames[0]['blur'])
        frame_idx = 0
        counter = 0
        for frame in best_frames:
            if int(frame['blur'])<int(min):
                frame_idx = counter
                min = frame['blur']
            counter+=1
        del best_frames[frame_idx]
    return best_frames


    
#use
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
   # blur_series, colorfulness_series = processColorAndBlur(yt, log = log)
    #print("-- Blur colorfulness series ok")
    serie,metadata = getImgseries(yt,domain,yolo,log=log)   
    tmp=metadata    
    #if process_color:
     #   serie = colorfulness_series
    average_cut = np.average(serie)    
    #frame_series = extractFramesFromSeries(max_frames, serie, log = log) #generazione serie di frame candidati
    print("-- Frame series ok")
    workdir = extractFrames(yt,serie, outputFolder, log = log) #estrazuibe dei frame nella work
    print("-- Extract frame ok")
    metadata = batchHistCompare(workdir,metadata,domain, process_faces = process_faces, corr = 0.5)#riconoscimento
    #get at least 5 scenes by improvement of frame correlation
    incr = 0.1
    frames_to_extract = n_max_frame
   # if log: print(metadata)
    lastScene = metadata[len(metadata)-1]
    lastScene_index = int(lastScene['scene'])
    if(len(metadata) < frames_to_extract):
        lastScene_index = frames_to_extract
        print("There are less than 5 scenes")
    while lastScene_index<frames_to_extract:
        metadata = batchHistCompare(workdir,tmp,domain, process_faces = process_faces, corr = 0.5+incr, log = log)
        incr = incr+0.1
        lastScene = metadata[len(metadata)-1]
        lastScene_index = int(lastScene['scene'])
    if log: print("5 filter")
    print(f"-- Selected metadata length: {len(metadata)}")
    with open(workdir+'/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4, separators=(',', ': '), sort_keys=True)
    #return generateThumb(workdir,frames_to_extract,log=log)
    return getColorResults(workdir, average_cut, frames_to_extract, log = log)







def predictFaces(image, log = False):
    if log: print('predictFaces: ',image)
    # Load the cascade
    t = os.getcwd()
    face_cascade = cv2.CascadeClassifier(t + '/haarcascade_frontalface_default.xml')
    if log: print('face_cascade')
    img = cv2.imread(image)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    predictions = {
                    "predictions": []
                    }
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
        predictions['predictions'].append(prediction)
    if log: print(json.dumps(predictions))
    return predictions



def executeSingle(outputFolder,**kwargs):
    yolo=YOLO()
   
    toProcess= []
    indexes = kwargs.get('indexes', None)
    domain=kwargs.get('domain')
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
        
    
def selectClasses(imgClass):
    
     food=['bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
                     'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','person']
     tech=['tvmonitor' ,'laptop', 'mouse' ,'remote', 'keyboard',
                        'cell', 'phone' ,'microwave','oven' ,'toaster']
     animal=['bird','cat','dog','horse','sheep','cow','elephant','bear',
                        'zebra','giraffe']
     car=['car','motorbike','truck','bus']
    
     for c in food:
        if c==imgClass:
            return 'food'
     for c in tech:
            if c==imgClass:
                return 'tech'
     for c in animal:
            if c==imgClass:
                return 'animal'
     for c in car:
            if c==imgClass:
                return 'car'
            
            
     return ''
           
      

def saveCsv():
    with open('youtube_dataset.csv','w')as file:
        writer = csv.writer(file)
        
        with open('youtube_boundingboxes.csv', 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    categorie=selectClasses(row[3])
                    if(categorie!=''):
                        row.append(categorie)
                        writer.writerow(row)
                    
    
def executeDataset(outputFolder,**kwargs):
     # yolo=YOLO()
      toProcess= []
      indexes = kwargs.get('indexes', None)
      domain=kwargs.get('domain')
      youtube=""
      tmp=""
      with open('youtube_dataset.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
              
                if(len(row)>0 and row[0]!=tmp) and row[3]!='knife':
                    if(domain==row[(len(row)-1)]):
                        toProcess.append('https://www.youtube.com/watch?v='+row[0])
                   
                    tmp=row[0]              
             
                
             
            
        
      if indexes is not None:
        toProcess = toProcess[indexes[0] : indexes[1]]
      
      yolo = YOLO()
      
      i=0
      for youtube in toProcess:
          try:
            print('Processing video '+youtube+' '+str(i)+'/'+str(len(toProcess)))
            start=time.time()
            processing(youtube, yolo, outputFolder, **kwargs)
            print('Processing video completed!!')
            print('Time: ',(time.time()-start))
            i=i+1
          except:
                logging.exception('ERROR')
                print(youtube + " cannot be processed.")          
              
    
    
    
   
            
        
def execute(sourcefile, outputFolder, **kwargs):
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, help='youtube url to be processed ')
    FLAGS = parser.parse_args()
    toProcess = []
    with open(sourcefile, 'rb') as f:
        videos = json.load(f)['videos']
    print("Full dim: "+str(len(videos)))
    for video in videos:
        toProcess.append('https://www.youtube.com/watch?v='+video['id'])
    indexes = kwargs.get('indexes', None)
    if indexes is not None:
        toProcess = toProcess[indexes[0] : indexes[1]]
    if FLAGS.video:
        print("Flag video: "+FLAGS.video)
        if FLAGS.video.startswith("https"):
            toProcess = [FLAGS.video]
        else:
            toProcess = ["https://www.youtube.com/watch?v="+FLAGS.video]
    print("dim toProcess: "+str(len(toProcess)))
    yolo = YOLO()
    i = 0
    for youtube in toProcess:
        try:
            print('Processing video '+youtube+' '+str(i)+'/'+str(len(toProcess)))
            processing(youtube, yolo, outputFolder, **kwargs)
            print('Processing video completed!!')
            i = i + 1
        except:
            logging.exception('ERROR')
            print(youtube + " cannot be processed.")          
            i = i + 1
   