"""
Emotion detection application flask server main page.
"""
import os
from flask import (
    Flask, render_template,
    Response, request, abort
    )
from werkzeug.utils import secure_filename
from track_video_web import VideoCamera
app = Flask(__name__)
app.config['UPLOAD_EXTENSIONS'] = ['.mp4', '.avi']
app.config['UPLOAD_PATH'] = 'form_uploads'

def gen_frames(videocamera_object):
    """
    method yiled frame from get_frame() method 
    after emotion detection.

    Parameters
    ----------
    None

    Return
    ------
    frame: yiled
        yield live streaming frame with emotion detection and 
        landmarks.
    """
    for frame in videocamera_object.get_frame():
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
        yield frame
        yield b'\r\n\r\n'
def get_form_data(request_var)-> str:
    """
    method will take the data from the form and after extracting return 
    the data.
    Parameters:
    ----------
    request_var: Request
        request send by the user from welcome.html page.
    Return:
    ------
    url: string
        return the data send by the user.

    """
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            abort(400)
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
        data = os.path.join(app.config['UPLOAD_PATH'], filename)
    else:
        data = request_var.form.get('url')
    return data

@app.route('/')
def index():
    """
    Index page emotion detcetion video streaming application.
    
    Parameters
    ----------
    None

    Return
    ------
        render template index.html.
    """
    return render_template('welcome.html')

@app.route('/video_feed', methods = ['POST'])
def video_feed():
    """
    method return response of yiled frame from gen_frames() method 
    after emotion detection.

    Parameters
    ----------
    None

    Return
    ------
    frame: Response
        return live streaming frame with emotion detection and 
        landmarks.
    """
    source = get_form_data(request)
    print(source)
    return Response(gen_frames(VideoCamera(source)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
