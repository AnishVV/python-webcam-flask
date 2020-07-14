from sys import stdout
from makeup_artist import Makeup_artist
import logging
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO
from camera import Camera
from utils import base64_to_pil_image, pil_image_to_base64


app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(stdout))
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True
socketio = SocketIO(app)
camera = Camera(Makeup_artist())


@socketio.on('input image', namespace='/test')
def test_message(input):
    input = input.split(",")[1]
    camera.enqueue_input(input)
    #camera.enqueue_input(base64_to_pil_image(input))


@socketio.on('connect', namespace='/test')
def test_connect():
    app.logger.info("client connected")


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen():
    """Video streaming generator function."""

    app.logger.info("starting to generate frames!")
    while True:
        frame = camera.get_frame() #pil_image_to_base64(camera.get_frame())
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


#########################################################
#########################################################
#########################################################\

@app.route('/detection_feed')
def detection_feed():
    resp = {}
    resp["hand_detection"] = True
    resp["posture"] = "slouching"
    return jsonify(resp)


if __name__ == '__main__':
    socketio.run(app)
