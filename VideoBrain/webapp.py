"""
Flask-Cors example
===================
This is a tiny Flask Application demonstrating Flask-Cors, making it simple
to add cross origin support to your flask app!
:copyright: (c) 2016 by Cory Dolphin.
:license:   MIT/X11, see LICENSE for more details.
"""
from PIL import Image
from flask import Flask, jsonify
from thumbnail_processor import processing, Predictor
from yolo import YOLO
import logging
try:
    from flask_cors import CORS  # The typical way to import flask-cors
except ImportError:
    # Path hack allows examples to be run without installation.
    import os
    parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.sys.path.insert(0, parentdir)

    from flask_cors import CORS
   # Needed to modify global copy of globvar
app = Flask('FlaskCorsAppBasedExample')
logging.basicConfig(level=logging.INFO)

# To enable logging for flask-cors,
logging.getLogger('flask_cors').level = logging.DEBUG

# One of the simplest configurations. Exposes all resources matching /api/* to
# CORS and allows the Content-Type header, which is necessary to POST JSON
# cross origin.
CORS(app, resources=r'/api/*')
predictor = Predictor.getInstance()

@app.route("/api/v1/processvideo/<id>")
def process_video(id):
    try:
        print ("process video")
        #youtube='https://www.youtube.com/watch?v='+id
        #print('Processing video '+youtube)
        #processing(youtube,yolo,5)
        try:
            file = 'out.jpg'
            image = Image.open(file)
            print('Image.opened: '+file)
            yolo = YOLO()
            yolo.detect_boxes(image)
            #print('Processing video completed '+youtube)
            return jsonify(success=True)
        except:
            print("inner error  cannot be processed.")
            return jsonify(success=False)
    except:
        print("outer error  cannot be processed.")
        return jsonify(success=False)


@app.route("/api/v1/users/create", methods=['POST'])
def create_user():
    """
        Since the path matches the regular expression r'/api/*', this resource
        automatically has CORS headers set.
        Browsers will first make a preflight request to verify that the resource
        allows cross-origin POSTs with a JSON Content-Type, which can be simulated
        as:
        $ curl --include -X OPTIONS http://127.0.0.1:5000/api/v1/users/create \
            --header Access-Control-Request-Method:POST \
            --header Access-Control-Request-Headers:Content-Type \
            --header Origin:www.examplesite.com
        >> HTTP/1.0 200 OK
        Content-Type: text/html; charset=utf-8
        Allow: POST, OPTIONS
        Access-Control-Allow-Origin: *
        Access-Control-Allow-Headers: Content-Type
        Access-Control-Allow-Methods: DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT
        Content-Length: 0
        Server: Werkzeug/0.9.6 Python/2.7.9
        Date: Sat, 31 Jan 2015 22:25:22 GMT
        $ curl --include -X POST http://127.0.0.1:5000/api/v1/users/create \
            --header Content-Type:application/json \
            --header Origin:www.examplesite.com
        >> HTTP/1.0 200 OK
        Content-Type: application/json
        Content-Length: 21
        Access-Control-Allow-Origin: *
        Server: Werkzeug/0.9.6 Python/2.7.9
        Date: Sat, 31 Jan 2015 22:25:04 GMT
        {
          "success": true
        }
    """
    try:
        youtube='https://www.youtube.com/watch?v=Zyzv9ZNYYmg'
        print('Processing video '+youtube)
        processing(youtube,getYolo(),5)
        print('Processing video completed '+youtube)
        return jsonify(success=True)
    except:
        print(youtube+" cannot be processed.")
        return jsonify(success=False)
   

@app.route("/api/exception")
def get_exception():
    """
        Since the path matches the regular expression r'/api/*', this resource
        automatically has CORS headers set.
        Browsers will first make a preflight request to verify that the resource
        allows cross-origin POSTs with a JSON Content-Type, which can be simulated
        as:
        $ curl --include -X OPTIONS http://127.0.0.1:5000/exception \
            --header Access-Control-Request-Method:POST \
            --header Access-Control-Request-Headers:Content-Type \
            --header Origin:www.examplesite.com
        >> HTTP/1.0 200 OK
        Content-Type: text/html; charset=utf-8
        Allow: POST, OPTIONS
        Access-Control-Allow-Origin: *
        Access-Control-Allow-Headers: Content-Type
        Access-Control-Allow-Methods: DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT
        Content-Length: 0
        Server: Werkzeug/0.9.6 Python/2.7.9
        Date: Sat, 31 Jan 2015 22:25:22 GMT
    """
    raise Exception("example")

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request. %s', e)
    return "An internal error occured", 500


if __name__ == "__main__":
    app.run(debug=True)