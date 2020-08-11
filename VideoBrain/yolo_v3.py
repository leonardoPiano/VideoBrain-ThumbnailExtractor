# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from keras import backend as K
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
import json

import sys
import argparse
from filters import smooth
from six.moves.urllib.parse import urlparse
import glob
import ntpath
import logging
import cv2
import time
import logging.config
import pafy
import numpy as np
from scipy import signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# To enable logging for flask-cors,
logging.getLogger('flask_cors').level = logging.DEBUG

class YOLO(object):
    __instance = None

    @staticmethod
    def getInstance():
        if YOLO.__instance == None:
            YOLO()
        else: logger.info("prediction models already loaded.")
        return YOLO.__instance 

    @staticmethod
    def resetInstance():
        logger.info("resetInstance()")
        K.clear_session();
        YOLO.__instance = None;

    def __init__(self,**kwargs):
        """ Virtually private constructor. """
        if YOLO.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            YOLO.__instance = self
            self.__dict__.update(self._defaults) # set up default values
            self.__dict__.update(kwargs) # and update with user overrides
            self.class_names = self._get_class()
            self.anchors = self._get_anchors()
            self.sess = K.get_session()
            self.boxes, self.scores, self.classes = self.generate()


    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"
    '''
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()
    '''
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        ##print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            #print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        #print(end - start)
        return image

    def detect_boxes(self, image):
        #extract boxes areas and labels
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        #print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        predictions = []
        for i, c in reversed(list(enumerate(out_classes))):
            objects = {}
            predicted_class = self.class_names[c]
            objects['class']=predicted_class
            box = out_boxes[i]
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            objects['box']= box.tolist()
            score = out_scores[i]
            objects['score']=score.tolist()
            #label = '{} {:.2f}'.format(predicted_class, score)
            area = (top-bottom)*(left-right)
            objects['area']=area.tolist()
            predictions.append(objects)

        end = timer()
        #print(end - start)
        #print(json.dumps(predictions))
        return predictions

    def close_session(self):
        self.sess.close()

    def extractFramesFromSeries(self,nframes,serie,video):
        logger.info("extractFramesFromSeries")
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
        logger.info("df average: "+str(average))
        max_peak_width = average
        if (max_peak_width<2):
            max_peak_width = 2
        peak_widths = np.arange(1, max_peak_width)
        logger.info("peak detection on average of derivative: "+str(peak_widths))
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
        logger.info("peak detection on average of derivative: "+str(peak_widths))
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
                        logger.info(e)
                        i = i + 1
                        logger.info("Cannot filter the array of zeros error in for")
                f_count = np.count_nonzero(array_of_zeros)
                incr = incr + 1
                #print ("f_count: "+str(f_count))
                if f_count>=nframes or incr==10:
                    break
        except Exception as e: 
            logger.info(e)
            logger.info("Cannot filter the array of zeros")
        print ('Average: '+str(average))
        #print (peak_indices)
        print ("local max found: "+str(peak_count))
        print ("Remove peak arount the average: "+str(np.count_nonzero(array_of_zeros)))
        r = np.empty(np.count_nonzero(array_of_zeros))
        ##logger.info(r)
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
        v_id = self.getVideoId(video)
        if not self.is_url(video):
            v_id = self.path_leaf(video)
        workdir = os.getcwd()+'/outputs1/workdirs'
        return r

    def getLocalMaxs(self,serie):
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

    def is_url(self,path):
        try:
            result = urlparse(path)
            return result.scheme and result.netloc and result.path
        except:
            return False

    def getVideoId(self,url):
        id = ''
        if (self.is_url(url)):
            list = url.split('?v=')
            id = list[len(list)-1]
        return id

    def estimate_blur(self,image, threshold=100):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_map = cv2.Laplacian(image, cv2.CV_64F)
        score = np.var(blur_map)
        return score, bool(score < threshold)

    def getBestVideo(self,isBest,video):
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

    def path_leaf(self,path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)

    def createFolder(self,id):
        newpath = os.getcwd()+'/outputs1/'+id 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        return newpath

    def createWorkDir(self,id):
        newpath = os.getcwd()+'/outputs1/workdirs/'+id 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        return newpath

    def compareHist(self,file1,file2,stat=cv2.HISTCMP_CORREL):
        target_im = cv2.imread(file1)
        target_hist = cv2.calcHist([target_im], [0], None, [256], [0, 256])
        comparing_im = cv2.imread(file2)
        comparing_hist = cv2.calcHist([comparing_im], [0], None, [256], [0, 256])
        diff = cv2.compareHist(target_hist, comparing_hist, stat)
        ##print(diff)
        return diff 

    def batchHistCompare(self, dir,corr=0.99):
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
                video_id = self.path_leaf(video)
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
                        fname = self.path_leaf(file)
                        ##print(file)
                        dist = self.compareHist(target,file)
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
                            logger.info('File Open Error! Try again!')
                        else:
                            metainfo['predictions'] = self.detect_boxes(image)
                        metadata.append(metainfo)
                        count = count + 1
                    except Exception as e: 
                        logger.info(e)
                        logger.error('video id {0} not processed image {1}.'.format(video_id,file))
                #logger.info(json.dumps(metadata))
            except Exception as e: 
                logger.info(e)
                logger.error('video id {0} not processed url {1}.'.format(video_id,video))
        return metadata
    
    def evaluateBlur(self,video_url,v=''):
        blur_series = []
        debug = True
        video = video_url
        #it returns the id if is the url of a youtube video
        v_id = self.getVideoId(video)

        if self.is_url(video):
            videoPafy = pafy.new(video)
            #video = videoPafy.getbest(preftype="mp4").url
            video = self.getBestVideo(True,videoPafy)
        else:
            v_id = v
        
        work_dir = os.getcwd()+'/outputs1/workdirs'
        workdir = self.createWorkDir(v_id)
            
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
                    #plotTimeSeries(blur_series)
                    return workdir

                start_prediction_time = time.time()
                blur_prediction = self.estimate_blur(frame)[0]
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

    def processBlur(self,video_url,v=''):
        logger.info("processBlur")
        blur_series = []
        debug = True
        video = video_url
        #it returns the id if is the url of a youtube video
        v_id = self.getVideoId(video)

        if self.is_url(video):
            videoPafy = pafy.new(video)
            #video = videoPafy.getbest(preftype="mp4").url
            video = self.getBestVideo(True,videoPafy)
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
                    logger.info('Can\'t read video data. Potential end of stream')
                    #plotTimeSeries(blur_series)
                    return blur_series

                start_prediction_time = time.time()
                blur_prediction = self.estimate_blur(frame)[0]
                if debug:
                    blur_series.append(blur_prediction)
                end_prediction_time = time.time()
                delay = end_prediction_time - start_prediction_time
                prediction_time = 'Blur estimation time: {0} ms blur: {1}, frame {2}'.format(delay,blur_prediction,frame_num)
                #logger.info('prediction time: '+prediction_time)

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

    def extractFrames(self,video_url,frame_series,v=''):
        logger.info("extractFrames ok")
        debug = True
        video = video_url
        #it returns the id if is the url of a youtube video
        v_id = self.getVideoId(video)
        logger.info("extractFrames video: "+video)
        if self.is_url(video):
            videoPafy = pafy.new(video)
            #video = videoPafy.getbest(preftype="mp4").url
            video = self.getBestVideo(True,videoPafy)
        else:
            v_id = v
        
        work_dir = os.getcwd()+'/outputs1/workdirs'
        workdir = self.createWorkDir(v_id)
            
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
                    logger.info('ExtractFrames Can\'t read video data. Potential end of stream')
                    return workdir

                start_prediction_time = time.time()
                blur_prediction = 0
            
                if (frame_num in frame_series) == True: 
                    blur_prediction = self.estimate_blur(frame)[0]
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
    
    def getBestResults(self,workDir,exactFrames=5):
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
        logger.info("***************************")
        #print(scenes)
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
            #logger.info("************** Best frame i="+str(i))
            #logger.info(best_frame)
            best_frames.append(best_frame)
        logger.info("*************************************")
        #logger.info(best_frames)
        best_frames = self.getOnlyNFrames(best_frames,exactFrames)
        with open(workDir+'/filtered_metadata.json', 'w') as f:
            json.dump(best_frames, f, indent=4, separators=(',', ': '), sort_keys=True)
        for file in best_frames:
            fname = self.path_leaf(file['file'])
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

    def getOnlyNFrames(self,best_frames,n=5):
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

    def processing(self,youtube,n_max_frame=5):
        logger.info("Processing youtube: "+youtube)
        #yolo = yolo_v3
        yt = youtube
        blur_series = self.processBlur(yt)
        logger.info("Blur series ok")
        #frame_series = plotTimeSeries(blur_series)
        max_frames = 50
        frame_series = self.extractFramesFromSeries(max_frames,blur_series,yt)
        #frame_series = getLocalMaxs(blur_series)
        #workdir = evaluateBlur('https://www.youtube.com/watch?v=Frnai8Dz9Tw')
        logger.info("Frame series ok")
        workdir = self.extractFrames(yt,frame_series)
        logger.info("Extract frame ok")
        logger.info('work dir is: {}'.format(workdir))
        metadata = self.batchHistCompare(workdir,0.5)
        logger.info("Metadata ok")
        #get at least 5 scenes by improvement of frame correlation
        incr = 0.1
        frames_to_extract = n_max_frame
        lastScene = metadata[len(metadata)-1]
        lastScene_index = int(lastScene['scene'])
        if(len(metadata)<frames_to_extract):
            lastScene_index = frames_to_extract
            logger.info("There are less than 5 scenes")
        while lastScene_index<frames_to_extract:
            metadata = self.batchHistCompare(workdir,0.5+incr)
            incr = incr+0.1
            lastScene = metadata[len(metadata)-1]
            lastScene_index = int(lastScene['scene'])
        logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        logger.info("5 filter")
        # now write output to a file
        with open(workdir+'/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4, separators=(',', ': '), sort_keys=True)
        return self.getBestResults(workdir,frames_to_extract)

