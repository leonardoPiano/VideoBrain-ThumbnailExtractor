import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import glob
import ntpath
import logging
import os

logger = logging.getLogger()

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def createFolder(id):
    newpath = '/Users/eugenio/Desktop/development/ImageClassification/classification/keras-yolo3/outputs/'+id 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath

def batch_img_detection():
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
    try:
        image = Image.open(path)
    except:
        print('Open Error! Try again!')
        return
    else:
        r_image = yolo.detect_image(image)
        #r_image.show()
        r_image.save(out)
    

FLAGS = None

if __name__ == '__main__':
    batch_img_detection()
    
