import sys
import argparse
from yolo import YOLO, detect_video
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
from scipy import signal
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# set the project root directory as the static folder, you can set others.
app = Flask(__name__, static_url_path= os.getcwd()+'/outputs1/workdirs')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# To enable logging for flask-cors,
logging.getLogger('flask_cors').level = logging.DEBUG

# One of the simplest configurations. Exposes all resources matching /api/* to
# CORS and allows the Content-Type header, which is necessary to POST JSON
# cross origin.
CORS(app, resources=r'/api/*')

class Predictor:
    """ This class manage the lodeing of the models as a singleton """
    __instance = None
    #public fields
    yolo = None
    lock = False

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

#analize the laplacian fame series
def extractFramesFromSeries(nframes,serie,video):
    print("extractFramesFromSeries")
    f = smooth(np.asarray(serie))
    s_av = np.average(f)
    f = f-s_av
    #f = signal.savgol_filter(f, 201, 3) # window size 51, polynomial order 3
    #df = np.gradient(f)
    df = smooth(np.gradient(f))
    df[df < 0] = 0
    y_coordinates = df # convert your 1-D array to a numpy array if it's not, otherwise omit this line
    #average of derivative of blur
    average = np.average(y_coordinates)
    df[df < average] = 0
    print("df average: "+str(average))
    max_peak_width = average
    if (max_peak_width<2):
        max_peak_width = 2
    peak_widths = np.arange(1, max_peak_width)
    print("peak detection on average of derivative: "+str(peak_widths))
    peak_indices = signal.find_peaks_cwt(y_coordinates, peak_widths)
    peak_count = len(peak_indices) # the number of peaks in the array
    print ("df average: "+str(average))
    print (peak_indices)
    print ("peak before filtering: "+str(peak_count))
    #array of zeros of lenght of derivative in order to filter peaks
    array_of_zeros = np.zeros(len(df),dtype=bool)
    i = 0
    #set to true the indexes that corresponds to a peak
    for x in np.nditer(array_of_zeros):
        array_of_zeros[i] = (i in peak_indices)
        i = i + 1
    #compress the peaks 
    array_peaks = df[array_of_zeros==True]
    #compute the average of the peaks compressed
    print ("Peaks array len: "+str(len(array_peaks)))
    average = np.average(array_peaks)
    print ("average Peaks array : "+str(average))
    max_peak_width = average
    if (max_peak_width<2):
        max_peak_width = 2
    peak_widths = np.arange(1, max_peak_width)
    print("peak detection on average of derivative: "+str(peak_widths))
    #peak_indices = signal.find_peaks_cwt(y_coordinates, peak_widths)
    peak_count = len(peak_indices) # the number of peaks in the array
    
    array_of_zeros = np.zeros(len(df),dtype=bool)
    print ("array_of_zeros: "+str(len(array_of_zeros)))
    try:
        f_count = 0
        incr = 1
        while True:
            i = 0
            #set to true the indexes that corresponds to a peak
            max = np.amax(f)
            print ("max: "+str(max))
            s_av = max - incr/10*max
            print ("cut off: "+str(s_av))
            f2 = f-s_av
            print ("Array of zezo len: "+str(len(array_of_zeros)))
            print ("f2 len: "+str(len(f2)))
            for x in np.nditer(array_of_zeros):
                try:
                    array_of_zeros[i] = (i in peak_indices) and (f2[i]>0)
                    i = i + 1
                except Exception as e: 
                    print(e)
                    i = i + 1
                    print("Cannot filter the array of zeros error in for")
            f_count = np.count_nonzero(array_of_zeros)
            incr = incr + 1
            #print ("f_count: "+str(f_count))
            if f_count>=nframes or incr==10:
                break
    except Exception as e: 
        print(e)
        print("Cannot filter the array of zeros")
    print ('Average: '+str(average))
    #print (peak_indices)
    print ("local max found: "+str(peak_count))
    print ("Remove peak arount the average: "+str(np.count_nonzero(array_of_zeros)))
    r = np.empty(np.count_nonzero(array_of_zeros))
    ##print(r)
    i = 0
    j = 0
    for x in np.nditer(peak_indices):
        ##print('idx i: '+str(i))
        ##print('Peak: '+str(peak_indices[i]))
        ##print('isZero: '+str(array_of_zeros[peak_indices[i]]))
        if(array_of_zeros[peak_indices[i]]):
            ##print('add value')
            if(j<np.count_nonzero(array_of_zeros)):
                r[j]=peak_indices[i]
                j = j + 1
        i = i + 1
    print ("Remove peak around the average: "+str(len(r)))
    print (r)
    v_id = getVideoId(video)
    if not is_url(video):
        v_id = path_leaf(video)
    workdir = os.getcwd()+'/outputs1/workdirs'
    return r

#analize the laplacian fame series
def plotTimeSeries(serie):
    f = np.asarray(serie)
    s_av = np.average(f)
    f = f-s_av
    #f = signal.savgol_filter(f, 201, 3) # window size 51, polynomial order 3
    df = np.gradient(f)
    df[df < 0] = 0
    y_coordinates = df # convert your 1-D array to a numpy array if it's not, otherwise omit this line
    #average of derivative of blur
    average = np.average(y_coordinates)
    df[df < average] = 0
    print("df average: "+str(average))
    max_peak_width = average
    peak_widths = np.arange(1, max_peak_width)
    #peak detection on average of derivative
    peak_indices = signal.find_peaks_cwt(y_coordinates, peak_widths)
    peak_count = len(peak_indices) # the number of peaks in the array
    print ("df average: "+str(average))
    #print (peak_indices)
    print ("peak before filtering: "+str(peak_count))
    #array of zeros of lenght of derivative in order to filter peaks
    array_of_zeros = np.zeros(len(df),dtype=bool)
    i = 0
    #set to true the indexes that corresponds to a peak
    for x in np.nditer(array_of_zeros):
        array_of_zeros[i] = (i in peak_indices)
        i = i + 1
    #compress the peaks 
    array_peaks = df[array_of_zeros==True]
    #compute the average of the peaks compressed
    print (len(array_peaks))
    average = np.average(array_peaks)
    max_peak_width = average
    peak_widths = np.arange(1, max_peak_width)
    peak_indices = signal.find_peaks_cwt(y_coordinates, peak_widths)
    peak_count = len(peak_indices) # the number of peaks in the array
    array_of_zeros = np.zeros(len(df),dtype=bool)
    i = 0
    #set to true the indexes that corresponds to a peak
    max = np.amax(f)
    s_av = max - 5/6*max
    print ("cut off: "+str(s_av))
    f = f-s_av
    for x in np.nditer(array_of_zeros):
        array_of_zeros[i] = (i in peak_indices) and (f[i]>0)
        i = i + 1
    print (average)
    print (peak_indices)
    print ("local max found: "+str(peak_count))
    print ("Remove peak arount the average: "+str(np.count_nonzero(array_of_zeros)))
    r = np.empty(np.count_nonzero(array_of_zeros))
    #print(r)
    i = 0
    j = 0
    for x in np.nditer(peak_indices):
        ##print('idx i: '+str(i))
        ##print('Peak: '+str(peak_indices[i]))
        ##print('isZero: '+str(array_of_zeros[peak_indices[i]]))
        if(array_of_zeros[peak_indices[i]]):
            ##print('add value')
            r[j]=peak_indices[i]
            j = j + 1
        i = i + 1
    print ("Remove peak arount the average: "+str(len(r)))
    print (r)
    return r

def getLocalMaxs(serie):
    f = np.asarray(serie)
    df = np.gradient(f)
    df[df < 0] = 0
    y_coordinates = df # convert your 1-D array to a numpy array if it's not, otherwise omit this line
    #average of derivative of blur
    average = np.average(y_coordinates)
    max_peak_width = average
    peak_widths = np.arange(1, max_peak_width)
    #peak detection on average of derivative
    peak_indices = signal.find_peaks_cwt(y_coordinates, peak_widths)
    peak_count = len(peak_indices) # the number of peaks in the array
    #array of zeros of lenght of derivative in order to filter peaks
    array_of_zeros = np.zeros(len(df),dtype=bool)
    i = 0
    #set to true the indexes that corresponds to a peak
    for x in np.nditer(array_of_zeros):
        array_of_zeros[i] = (i in peak_indices)
        i = i + 1
    #compress the peaks 
    array_peaks = df[array_of_zeros==True]
    #compute the average of the peaks compressed
    average = np.average(array_peaks)
    max_peak_width = average
    peak_widths = np.arange(1, max_peak_width)
    peak_indices = signal.find_peaks_cwt(y_coordinates, peak_widths)
    peak_count = len(peak_indices) # the number of peaks in the array
    array_of_zeros = np.zeros(len(df),dtype=bool)
    i = 0
    #set to true the indexes that corresponds to a peak
    for x in np.nditer(array_of_zeros):
        array_of_zeros[i] = (i in peak_indices)
        i = i + 1
    print (len(peak_indices))
    return peak_indices

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

def getBestVideo(isBest,video):
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

def createWorkDir(id):
    newpath = os.getcwd()+'/outputs1/workdirs/'+id 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath


def compareHist(file1,file2,stat=cv2.HISTCMP_CORREL):
    target_im = cv2.imread(file1)
    target_hist = cv2.calcHist([target_im], [0], None, [256], [0, 256])
    comparing_im = cv2.imread(file2)
    comparing_hist = cv2.calcHist([comparing_im], [0], None, [256], [0, 256])
    diff = cv2.compareHist(target_hist, comparing_hist, stat)
    ##print(diff)
    return diff 

def batchHistCompare(dir,yolo,corr=0.99):
    #it includes image detection
    #yolo = YOLO()
    #param video 
    #param blur threshold
    #param frame correlation
    b_limit = 100
    f_dist = corr
    metadata = []
    #videos = glob.glob('/Users/eugenio/Desktop/development/ImageClassification/YOLO/devicehive-video-analysis/videos/*')
    videos = ['/Users/eugenio/Desktop/development/ImageClassification/YOLO/devicehive-video-analysis/videos/1RIO2OqsyIc']
    if dir: videos = [dir]
    toEnd = len(videos)
    #print('videos to process: '+str(toEnd))
    for video in videos:
        try:
            #print('Remaining videos to process: '+str(toEnd))
            logger.info(video)
            video_id = path_leaf(video)
            #work_dir = createFolder(video_id)
            #logger.info(work_dir)
            files = glob.glob(video+'/*')
            files.sort(key=os.path.getmtime)
            #logger.info(files)
            count = 0
            scene_counter = 0
            target = files[0]
            for file in files:
                try:
                    fname = path_leaf(file)
                    ##print(file)
                    dist = compareHist(target,file)
                    target = file
                    isChanged = " no scene change."
                    if (dist<f_dist): 
                        isChanged = " THIS IS A NEW SCENE."
                        scene_counter = scene_counter +1
                    #print('Image {0}, {2} compare {1}. {3}'.format(count,dist,path_leaf(file),isChanged))
                    metainfo = {}
                    metainfo['file'] = file
                    metainfo['frame'] = fname.split('_')[1]
                    metainfo['blur'] = fname.split('_')[2].split('.')[0]
                    metainfo['scene'] = scene_counter
                    metainfo['corr'] = dist
                    try:
                        image = Image.open(file)
                    except:
                        print('File Open Error! Try again!')
                    else:
                        metainfo['predictions'] = yolo.detect_boxes(image)
                    metadata.append(metainfo)
                    count = count + 1
                except Exception as e: 
                    print(e)
                    logger.error('video id {0} not processed image {1}.'.format(video_id,file))
            #print(json.dumps(metadata))
        except Exception as e: 
            print(e)
            logger.error('video id {0} not processed url {1}.'.format(video_id,video))
    return metadata

def batch_img_detection():
    #param directory to be processed
    yolo = YOLO()
    #videos = glob.glob('/Users/eugenio/Desktop/development/ImageClassification/YOLO/devicehive-video-analysis/videos/*')
    videos = glob.glob('/Users/eugenio/Desktop/development/ImageClassification/YOLO/devicehive-video-analysis/videos_processed/*')
    toEnd = len(videos)
    logger.info('videos to process: '+str(toEnd))
    count = 0
    #logger.info(videos)
    for video in videos:
        try:
            logger.info('Remaining videos to process: '+str(toEnd))
            logger.info(video)
            video_id = path_leaf(video)
            work_dir = createFolder(video_id)
            logger.info(work_dir)
            files = glob.glob(video+'/*')
            #logger.info(files)
            count = 0
            for file in files:
                try:
                    logger.info(file)
                    detect_img(yolo, file, work_dir+'/YOLOv3_'+path_leaf(file))
                except:
                    logger.error('video id {0} not processed image {1}.'.format(video_id,file))
            count = count + 1

        except:
            logger.error('video id {0} not processed url {1}.'.format(video_id,video))
    yolo.close_session()

def detect_img(yolo,path,out='out.jpg'):
    #this function draw the boxes around the detected objects
    try:
        image = Image.open(path)
    except:
        #print('Open Error! Try again!')
        return
    else:
        r_image = yolo.detect_image(image)
        #r_image.show()
        r_image.save(out)
    
def evaluateBlur(video_url,v=''):
    blur_series = []
    debug = True
    video = video_url
    #it returns the id if is the url of a youtube video
    v_id = getVideoId(video)

    if is_url(video):
        videoPafy = pafy.new(video)
        #video = videoPafy.getbest(preftype="mp4").url
        video = getBestVideo(True,videoPafy)
    else:
        v_id = v
    
    work_dir = os.getcwd()+'/outputs1/workdirs'
    workdir = createWorkDir(v_id)
        
    #print('Video id: {} Workdir {}'.format(v_id,workdir))


    cam = cv2.VideoCapture(video)
    if not cam.isOpened():
        raise IOError('Can\'t open "{}"'.format('Yolo2Model'))

    frame_num = 0
    start_time = time.time()
    fps = 0
    max = 0
    try:
        while True:
            ret, frame = cam.read()

            if not ret:
                #print('Can\'t read video data. Potential end of stream')
                plotTimeSeries(blur_series)
                return workdir

            start_prediction_time = time.time()
            blur_prediction = estimate_blur(frame)[0]
            if debug:
                blur_series.append(blur_prediction)
            if (blur_prediction > max): 
                max = blur_prediction
                cv2.imwrite(work_dir+'/{0}/frame_{1}_{2}.jpg'.format(v_id,frame_num,max), frame)
            end_prediction_time = time.time()
            delay = end_prediction_time - start_prediction_time
            prediction_time = 'Blur estimation time: {0} ms blur: {1}, max blur {2}'.format(delay,blur_prediction,max)
            #print('prediction time: '+prediction_time)

            end_time = time.time()
            fps = fps * 0.9 + 1/(end_time - start_time) * 0.1
            start_time = end_time

            # Draw additional info
            frame_info = 'Frame: {0}, FPS: {1:.2f}, Score: {2}'.format(frame_num, fps, blur_prediction)
            cv2.putText(frame, frame_info, (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            key = cv2.waitKey(1) & 0xFF

            # Exit
            if key == ord('q'):
                break

            # Take screenshot
            if key == ord('s'):
                cv2.imwrite('frame_{}.jpg'.format(time.time()), frame)

            frame_num += 1

    finally:
        cam.release()

def processBlur(video_url,v=''):
    print("processBlur")
    blur_series = []
    debug = True
    video = video_url
    #it returns the id if is the url of a youtube video
    v_id = getVideoId(video)

    if is_url(video):
        videoPafy = pafy.new(video)
        #video = videoPafy.getbest(preftype="mp4").url
        video = getBestVideo(True,videoPafy)
    else:
        v_id = v

    cam = cv2.VideoCapture(video)
    if not cam.isOpened():
        raise IOError('Can\'t open "{}"'.format('Yolo2Model'))

    frame_num = 0
    start_time = time.time()
    fps = 0
    max = 0
    try:
        while True:
            ret, frame = cam.read()

            if not ret:
                print('Can\'t read video data. Potential end of stream')
                #plotTimeSeries(blur_series)
                return blur_series

            start_prediction_time = time.time()
            blur_prediction = estimate_blur(frame)[0]
            if debug:
                blur_series.append(blur_prediction)
            end_prediction_time = time.time()
            delay = end_prediction_time - start_prediction_time
            prediction_time = 'Blur estimation time: {0} ms blur: {1}, frame {2}'.format(delay,blur_prediction,frame_num)
            #print('prediction time: '+prediction_time)

            end_time = time.time()
            fps = fps * 0.9 + 1/(end_time - start_time) * 0.1
            start_time = end_time

            key = cv2.waitKey(1) & 0xFF

            # Exit
            if key == ord('q'):
                break

            frame_num += 1

    finally:
        cam.release()

def extractFrames(video_url,frame_series,v=''):
    print("extractFrames ok")
    debug = True
    video = video_url
    #it returns the id if is the url of a youtube video
    v_id = getVideoId(video)
    print("extractFrames video: "+video)
    if is_url(video):
        videoPafy = pafy.new(video)
        #video = videoPafy.getbest(preftype="mp4").url
        video = getBestVideo(True,videoPafy)
    else:
        v_id = v
    
    work_dir = os.getcwd()+'/outputs1/workdirs'
    workdir = createWorkDir(v_id)
        
    #print('Video id: {} Workdir {}'.format(v_id,workdir))


    cam = cv2.VideoCapture(video)
    if not cam.isOpened():
        raise IOError('ExtractFrames Can\'t open "{}"'.format('Yolo2Model'))

    frame_num = 0
    start_time = time.time()
    fps = 0

    try:
        while True:
            ret, frame = cam.read()

            if not ret:
                print('ExtractFrames Can\'t read video data. Potential end of stream')
                return workdir

            start_prediction_time = time.time()
            blur_prediction = 0
           
            if (frame_num in frame_series) == True: 
                blur_prediction = estimate_blur(frame)[0]
                cv2.imwrite(work_dir+'/{0}/localMaxFrame_{1}_{2}.jpg'.format(v_id,frame_num,blur_prediction), frame)
            end_prediction_time = time.time()
            delay = end_prediction_time - start_prediction_time
            prediction_time = 'Blur estimation time: {0} ms blur: {1} , frame {2}'.format(delay,blur_prediction,frame_num)
            #print('prediction time: '+prediction_time)

            end_time = time.time()
            fps = fps * 0.9 + 1/(end_time - start_time) * 0.1
            start_time = end_time

            # Draw additional info
            frame_info = 'Frame: {0}, FPS: {1:.2f}, Score: {2}'.format(frame_num, fps, blur_prediction)
            cv2.putText(frame, frame_info, (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            key = cv2.waitKey(1) & 0xFF

            # Exit
            if key == ord('q'):
                break

            # Take screenshot
            if key == ord('s'):
                cv2.imwrite('frame_{}.jpg'.format(time.time()), frame)

            frame_num += 1

    finally:
        cam.release()
   
def getBestResults(workDir,exactFrames=5):
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
    print("***************************")
    print(scenes)
    #best frame for scene max object area if blur > 1000
    best_frames = []
    for i in range(to+1):
        frames = scenes[i]
        max_blur = 100
        ref_blur = 100
        max_score = 0.5
        best_frame = scenes[i][0]
        for frame in frames:
            if (int(frame['blur'])>max_blur):
                max_blur = int(frame['blur'])
                best_frame = frame
        for frame in frames:
            if (int(frame['blur'])>ref_blur):
                #seach for max object score 
                if (len(frame['predictions'])>0):
                    max_area = 0
                    predictions = frame['predictions']
                    for prediction in predictions:
                        if (float(prediction['score'])>max_score):
                            if(int(prediction['area'])>max_area):
                                max_area = int(prediction['area'])
                                best_frame = frame
        #print("************** Best frame i="+str(i))
        #print(best_frame)
        best_frames.append(best_frame)
    print("*************************************")
    print(best_frames)
    best_frames = getOnlyNFrames(best_frames,exactFrames)
    with open(workDir+'/filtered_metadata.json', 'w') as f:
        json.dump(best_frames, f, indent=4, separators=(',', ': '), sort_keys=True)
    for file in best_frames:
        fname = path_leaf(file['file'])
        new_name = workDir+'selected_'+fname
        os.rename(file['file'], new_name)
        #print(new_name)
    #print("Blob")
    images = glob.glob(workDir+'*')
    for filename in images:
        if filename.startswith(workDir+'localMaxFrame') or filename.endswith('.json'):
            #print(filename)
            os.unlink(filename)
    return best_frames

def getOnlyNFrames(best_frames,n=5):
    while len(best_frames)>n:
        #remove min
        min = int(best_frames[0]['blur'])
        frame_idx = 0
        counter = 0
        for frame in best_frames:
            if int(frame['blur'])<int(min):
                frame_idx = counter
                min = frame['blur']
        del best_frames[frame_idx]
    return best_frames

def processing(youtube,yolo_v3,n_max_frame=5):
    print("Processing at: "+os.getcwd())
    yolo = yolo_v3
    yt = youtube
    blur_series = processBlur(yt)
    print("Blur series ok")
    #frame_series = plotTimeSeries(blur_series)
    max_frames = 50
    frame_series = extractFramesFromSeries(max_frames,blur_series,yt)
    #frame_series = getLocalMaxs(blur_series)
    #workdir = evaluateBlur('https://www.youtube.com/watch?v=Frnai8Dz9Tw')
    print("Frame series ok")
    workdir = extractFrames(yt,frame_series)
    print("Extract frame ok")
    print('work dir is: {}'.format(workdir))
    metadata = batchHistCompare(workdir,yolo,0.5)
    print("Metadata ok")
    #get at least 5 scenes by improvement of frame correlation
    incr = 0.1
    frames_to_extract = n_max_frame
    lastScene = metadata[len(metadata)-1]
    lastScene_index = int(lastScene['scene'])
    if(len(metadata)<frames_to_extract):
        lastScene_index = frames_to_extract
        print("There are less than 5 scenes")
    while lastScene_index<frames_to_extract:
        metadata = batchHistCompare(workdir,yolo,0.5+incr)
        incr = incr+0.1
        lastScene = metadata[len(metadata)-1]
        lastScene_index = int(lastScene['scene'])
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print("5 filter")
    # now write output to a file
    with open(workdir+'/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4, separators=(',', ': '), sort_keys=True)
    return getBestResults(workdir,frames_to_extract)

@app.route("/api/v1/processvideo/<id>")
def process_video(id):
    if Predictor.getInstance().lock:
        print('lock: '+str(Predictor.getInstance().lock))
    youtube = 'https://www.youtube.com/watch?v='+id
    Predictor.getInstance().lock = True
    try:
        os.system('python video_processor.py --video '+youtube)
        return jsonify(success=True)
    except Exception as e: 
        print(e)
        print(youtube+" cannot be processed.")
        Predictor.getInstance().lock = False
        return jsonify(success=False)

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request. %s', e)
    return "An internal error occured", 500

'''
if __name__ == "__main__":
    app.run(debug=True,threaded=True)
'''

from http.server import BaseHTTPRequestHandler,HTTPServer
from urllib.parse import urlparse
import json
import string,cgi,time
from os import curdir, sep

PORT_NUMBER = 8080
yolo_ = YOLO()

'''
class MyHandler(BaseHTTPRequestHandler):
    
    def do_GET(self):
        try:
            if self.path.endswith(".jpg"):
                f = open(curdir + sep + self.path, 'rb')
                self.send_response(200)
                self.send_header('Content-type',    'text/html')
                self.end_headers()
                self.wfile.write(f.read())
                f.close()
                return

        except IOError:
            self.send_error(404,'File Not Found: %s' % self.path)

'''


#This class will handles any incoming request from
#the browser 
class myHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            if self.path.endswith(".jpg"):
                print('self.path.endswith(".jpg"): '+self.path)
                print('/Users/eugenio/Desktop/development/ImageClassification/classification/keras-yolo3/outputs1/workdirs' + self.path)
                f = open('/Users/eugenio/Desktop/development/ImageClassification/classification/keras-yolo3/outputs1/workdirs' + self.path, 'rb')
                self.send_response(200)
                self.send_header('Content-type',    'image/jpeg')
                self.end_headers()
                self.wfile.write(f.read())
                f.close()
                return
        except IOError:
            self.send_error(404,'File Not Found: %s' % self.path)
            return

        self.send_response(200)
        self.send_header('Content-type','application/json')
        self.end_headers()
        print(self.headers)
        if self.path: 
            print(self.path)
        query = urlparse(self.path).query
        result = {}
        if query:
            print(query)
            query_components = dict(qc.split("=") for qc in query.split("&"))
            id = query_components.get('id')
            print(query_components.get('id'))
            result=processing('https://www.youtube.com/watch?v='+id,yolo_)
        self.wfile.write(json.dumps(result).encode())
        return

try:
	#Create a web server and define the handler to manage the
	#incoming request
	server = HTTPServer(('', PORT_NUMBER), myHandler)
	print('Started httpserver on port ' , PORT_NUMBER)
	
	#Wait forever for incoming htto requests
	server.serve_forever()

except KeyboardInterrupt:
	print('^C received, shutting down the web server')
	server.socket.close()
