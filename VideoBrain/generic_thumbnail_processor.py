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

#analize the laplacian fame series
def extractFramesFromSeries(nframes,serie, log = False):
    if log: print("extractFramesFromSeries")
    
    f = smooth(np.asarray(serie))
    s_av = np.average(f)
    f = f-s_av
    df = smooth(np.gradient(f))
    df[df < 0] = 0
    y_coordinates = df # convert your 1-D array to a numpy array if it's not, otherwise omit this line
    #average of derivative of blur
    average = np.average(y_coordinates)
    df[df < average] = 0
    if log: print("df average: "+str(average))
    max_peak_width = average
    if (max_peak_width<2):
        max_peak_width = 2
    peak_widths = np.arange(1, max_peak_width)
    if log: print("peak detection on average of derivative: "+str(peak_widths))
    peak_indices = signal.find_peaks_cwt(y_coordinates, peak_widths)
    peak_count = len(peak_indices) # the number of peaks in the array
    if log: 
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
#    plt.figure(0)
#    plt.plot(array_of_zeros*100) 
    #compress the peaks 
    array_peaks = df[array_of_zeros==True]
    #compute the average of the peaks compressed
    if log: print ("Peaks array len: "+str(len(array_peaks)))
    average = np.average(array_peaks)
    if log: print ("average Peaks array : "+str(average))
    max_peak_width = average
    if (max_peak_width<2):
        max_peak_width = 2
    peak_widths = np.arange(1, max_peak_width)
    if log: print("peak detection on average of derivative: "+str(peak_widths))
    #peak_indices = signal.find_peaks_cwt(y_coordinates, peak_widths)
    peak_count = len(peak_indices) # the number of peaks in the array   
    array_of_zeros = np.zeros(len(df),dtype=bool)
    if log: print ("array_of_zeros: "+str(len(array_of_zeros)))
    try:
        f_count = 0
        incr = 1
        while True:
            i = 0
            #set to true the indexes that corresponds to a peak
            max = np.amax(f)
            if log: print ("max: "+str(max))
            s_av = max - incr/10*max
            if log: print ("cut off: "+str(s_av))
            f2 = f-s_av
            if log: print ("Array of zezo len: "+str(len(array_of_zeros)))
            if log: print ("f2 len: "+str(len(f2)))
            for x in np.nditer(array_of_zeros):
                try:
                    array_of_zeros[i] = (i in peak_indices) and (f2[i]>0)
                    i = i + 1
                except Exception as e: 
                    if log: print(e)
                    i = i + 1
                    if log: print("Cannot filter the array of zeros error in for")
#            plt.figure(1)
#            plt.plot(array_of_zeros*100) 
            f_count = np.count_nonzero(array_of_zeros)
            incr = incr + 1
            #print ("f_count: "+str(f_count))
            if f_count>=nframes or incr==10:
                break
    except Exception as e: 
        print(e)
        print("Cannot filter the array of zeros")
    if log: print ('Average: '+str(average))
    if log: print ("local max found: "+str(peak_count))
    if log: print ("Remove peak arount the average: "+str(np.count_nonzero(array_of_zeros)))
    r = np.empty(np.count_nonzero(array_of_zeros))
   
    i = 0
    j = 0
    for x in np.nditer(peak_indices):
        if(array_of_zeros[peak_indices[i]]):
            if(j<np.count_nonzero(array_of_zeros)):
                r[j]=peak_indices[i]
                j = j + 1
        i = i + 1
    if log: 
        print ("Remove peak around the average: "+str(len(r)))
        print (r)
    
    return r

#used
#analize the laplacian fame series
def plotTimeSeries(serie, log = False):
    f = np.asarray(serie)
    s_av = np.average(f)
    f = f-s_av
    df = np.gradient(f)
    df[df < 0] = 0
    plt.plot(f) # plotting by columns
    y_coordinates = df # convert your 1-D array to a numpy array if it's not, otherwise omit this line
    #average of derivative of blur
    average = np.average(y_coordinates)
    df[df < average] = 0
    if log: print("df average: "+str(average))
    plt.plot(df) # plotting by columns
    max_peak_width = average
    peak_widths = np.arange(1, max_peak_width)
    #peak detection on average of derivative
    peak_indices = signal.find_peaks_cwt(y_coordinates, peak_widths)
    peak_count = len(peak_indices) # the number of peaks in the array
    if log: 
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
    plt.plot(array_of_zeros) 
    #compress the peaks 
    array_peaks = df[array_of_zeros==True]
    #compute the average of the peaks compressed
    if log: print (len(array_peaks))
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
    if log: print ("cut off: "+str(s_av))
    f = f-s_av
    for x in np.nditer(array_of_zeros):
        array_of_zeros[i] = (i in peak_indices) and (f[i]>0)
        i = i + 1
    plt.plot(array_of_zeros*100) 
    if log: 
        print (average)
        print (peak_indices)
        print ("local max found: "+str(peak_count))
        print ("Remove peak arount the average: "+str(np.count_nonzero(array_of_zeros)))
    r = np.empty(np.count_nonzero(array_of_zeros))
    if log: print(r)
    i = 0
    j = 0
    for x in np.nditer(peak_indices):
        if(array_of_zeros[peak_indices[i]]):
            ##print('add value')
            r[j]=peak_indices[i]
            j = j + 1
        i = i + 1
    if log: 
        print ("Remove peak arount the average: "+str(len(r)))
        print (r)
    plt.show()
    return r

def plot2TimeSeries(serie1,serie2):
    max_colofulness = np.amax(serie2)
    print('Max element from Numpy Array : ', max_colofulness)
    colofulness_average = np.average(serie2)
    print('Average value from Numpy Array : ', colofulness_average)
    max_blur = np.amax(serie1)
    print('Max element from Blur Array : ', max_blur)
    blur_average = np.average(serie1)
    print('Average value from Blur Array : ', blur_average)
    f = np.asarray(serie1)-blur_average
    f[f < 0] = 0
    f_normed = f / f.max(axis=0)
    f2 = np.asarray(serie2)-colofulness_average
    f2[f2 < 0] = 0
    f2_normed = f2 / f2.max(axis=0)
    df = np.gradient(f)
    df2 = np.gradient(f2)

    plt.plot(f_normed) # plotting by columns
    plt.plot(f2_normed,color='green') # plotting by columns
    #average of derivative of blur
    df[df < np.average(df)] = 0
    df2[df2 < np.average(df2)] = 0
    plt.show()

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
   
    return diff 

def batchHistCompare(dir, yolo, domain, process_faces = True, corr=0.99, log = False):
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
                        image = Image.open(file)
                        
                      
                        if log: print('Image.opened: '+file)
                    except:
                        print('File Open Error! Try again!')
                    else:
                        if log: print('Try to make a prediction on faces.')
                        if process_faces:
                            predictions = predictFaces(file)
                            metainfo['predictions'] = predictions['predictions']
                        else:                            
                            metainfo['predictions'] = yolo.detect_img(image,domain)                         
                    metadata.append(metainfo)
                    count = count + 1
                except:
                    logger.error(f'video id {video_id} not processed image {file}.', exc_info=True)
        except:
            logger.error(f'video id {video_id} not processed url {video}.')
    return metadata
    #this function draw the boxes around the detected objects


def processColorAndBlur(video_url,v='', log = False):
    if log: print("processColorAndBlur")
    colorfulness_series = []
    blur_series = []
    debug = True
    video = video_url
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
    try:
        while True:
            ret, frame = cam.read()
          
            if not ret:
                if log: print('Can\'t read video data. Potential end of stream')
                return blur_series,colorfulness_series
            blur_prediction = estimate_blur(frame)[0]#blur score
       
            colorfulness_prediction = image_colorfulness(frame)
            if debug:
             
                blur_series.append(blur_prediction)
                colorfulness_series.append(colorfulness_prediction)
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
            if (frame_num in frame_series) == True: 
                blur_prediction = estimate_blur(frame)[0]#blur frames
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
        if ('\localMaxFrame' in filename) or filename.endswith('.json'):
            os.unlink(filename)
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
    blur_series, colorfulness_series = processColorAndBlur(yt, log = log)
    print("-- Blur colorfulness series ok")
    serie = blur_series

    if process_color:
        serie = colorfulness_series
    average_cut = np.average(serie)    
    frame_series = extractFramesFromSeries(max_frames, serie, log = log)   
    print("-- Frame series ok")
    workdir = extractFrames(yt,frame_series, outputFolder, log = log)
    print("-- Extract frame ok")
    metadata = batchHistCompare(workdir, yolo,domain, process_faces = process_faces, corr = 0.5)
    #get at least 5 scenes by improvement of frame correlation
    incr = 0.1
    frames_to_extract = n_max_frame
    if log: print(metadata)
    lastScene = metadata[len(metadata)-1]
    lastScene_index = int(lastScene['scene'])
    if(len(metadata) < frames_to_extract):
        lastScene_index = frames_to_extract
        print("There are less than 5 scenes")
    while lastScene_index<frames_to_extract:
        metadata = batchHistCompare(workdir, yolo,domain, process_faces = process_faces, corr = 0.5+incr, log = log)
        incr = incr+0.1
        lastScene = metadata[len(metadata)-1]
        lastScene_index = int(lastScene['scene'])
    if log: print("5 filter")
    print(f"-- Selected metadata length: {len(metadata)}")
    with open(workdir+'/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4, separators=(',', ': '), sort_keys=True)
    return getColorResults(workdir, average_cut, frames_to_extract, log = log)



def image_colorfulness(image):
      # split the image into its respective RGB components
	(B, G, R) = cv2.split(image.astype("float"))
 
	# compute rg = R - G
	rg = np.absolute(R - G)
    
	# compute yb = 0.5 * (R + G) - B
	yb = np.absolute(0.5 * (R + G) - B)
 
	# compute the mean and standard deviation of both `rg` and `yb`
	(rbMean, rbStd) = (np.mean(rg), np.std(rg))
	(ybMean, ybStd) = (np.mean(yb), np.std(yb))
 
	# combine the mean and standard deviations
	stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
	meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
 
	# derive the "colorfulness" metric and return it
	return stdRoot + (0.3 * meanRoot)



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
            start=time.time()
            print('Processing video '+youtube)
            processing(youtube, yolo, outputFolder, **kwargs)
            print('Processing video completed!!')
            print('Time', time.time()-start)
        except:
            logging.exception('ERROR')
            print(youtube + " cannot be processed.")          
        
    
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
   