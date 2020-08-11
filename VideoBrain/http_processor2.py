import sys
import argparse
from yolo_v3 import YOLO
from six.moves.urllib.parse import urlparse
import ntpath
import logging
import os
import time
import logging.config
import json
from http.server import BaseHTTPRequestHandler,HTTPServer
from urllib.parse import urlparse
import json
import string,cgi,time
from os import curdir, sep
import threading
# source bin/activate
PORT_NUMBER = 8080

in_process=''
status='READY' #READY, ERROR??, PROCESSING
#This class will handles any incoming request from
#the browser 

def setup_custom_logger(name):
    #formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler = logging.StreamHandler()
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    #logger.setFormatter(formatter)
    logger.addHandler(screen_handler)
    return logger

logger = setup_custom_logger('http-processor')

class myHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Credentials', 'true')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-type")

    def do_GET(self):
        try:
            if self.path.endswith(".jpg"):
                logger.info('http_processor - self.path.endswith(".jpg"): '+self.path)
                logger.info('http_processor - '+curdir+sep+'outputs1/workdirs' + self.path)
                f = open(curdir+sep+'outputs1/workdirs' + self.path, 'rb')
                self.send_response(200)
                self.send_header('Content-type','image/jpeg')
                self.send_header('Access-Control-Allow-Credentials', 'true')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(f.read())
                f.close()
                return
        except Exception as e: 
            logger.info('http_processor - '+str(e))
            self.send_error(404,'File Not Found: %s' % self.path)
            return
        self.send_response(200)
        self.send_header('Content-type','application/json')
        self.send_header('Access-Control-Allow-Credentials', 'true')
        self.send_header('Access-Control-Allow-Origin', '*')
        global in_process
        global status
        logger.info('http_processor - status: '+status)
        logger.info('http_processor - id: '+in_process)
        if status!='READY':
            logger.info('http_processor - status: NOT READY')
            proc={
                "status":status,
                "id":in_process
            }
            self.end_headers()
            self.wfile.write(json.dumps(proc).encode())
            return
        self.end_headers()
        #logger.info(self.headers)
        if self.path: 
            logger.info(self.path)
        query = urlparse(self.path).query
        result = {}
        if query:
            logger.info(query)
            query_components = dict(qc.split("=") for qc in query.split("&"))
            id = query_components.get('id')
            logger.info(query_components.get('id'))
            try:
                fh = open(curdir+sep+'outputs1/workdirs/'+id+'/metadata.json', 'rb')
                self.wfile.write(fh.read())
                fh.close()
                return
            except Exception as e: 
                logger.info(e)
            in_process=id
            def worker(id):
                try:
                    logger.info('worker started')
                    global in_process
                    global status
                    in_process=id
                    status = 'PROCESSING'
                    result = YOLO.getInstance().processing('https://www.youtube.com/watch?v='+id)
                    for r in result:
                        r['file']='/'+id+'/selected_'+YOLO.getInstance().path_leaf(r['file'])
                        logger.info(r['file'])
                    in_process=''
                    status = 'READY'
                    with open(curdir+sep+'outputs1/workdirs/'+id+'/metadata.json', 'w') as f:
                        json.dump(result, f, indent=4, separators=(',', ': '), sort_keys=True)
                    YOLO.resetInstance()
                except Exception as e: 
                    logger.info(e)
                    status = 'READY'
                    in_process=''
                    YOLO.resetInstance()
                    return
            t = threading.Thread(target=worker, args=(id,))
            t.start()
            proc={
                "status":'PROCESSING',
                "id":in_process
            }
            self.wfile.write(json.dumps(proc).encode())
            return
        self.wfile.write(json.dumps(result).encode())
        return

try:
	#Create a web server and define the handler to manage the
	#incoming request
	server = HTTPServer(('', PORT_NUMBER), myHandler)
	logger.info('Started httpserver on port '+str(PORT_NUMBER))
	
	#Wait forever for incoming htto requests
	server.serve_forever()

except KeyboardInterrupt:
	logger.info('^C received, shutting down the web server')
	server.socket.close()

#Python3 on OSX: /usr/local/Cellar/python/3.6.5_1/Frameworks/Python.framework/Versions/3.6/bin/python3.6