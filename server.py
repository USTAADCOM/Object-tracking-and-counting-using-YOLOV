"""
Main module of flask API.
"""
# Third party modules
from typing import Any
from functools import wraps
import os
import datetime
import imghdr
import gofile as go
from dotenv import load_dotenv
from flask import (
    Flask, request,
    json, make_response, Response
)
from flask_cors import CORS, cross_origin
from PIL import Image
# Module
import detect_image_objects
app = Flask(__name__)
cors = CORS(app)
app.config['PROPAGATE_EXCEPTIONS'] = True
upload_folder = 'api_uploads'
os.makedirs(upload_folder, exist_ok = True)
def authorize(token: str)-> bool:
    """
    method take header token as input and check valid ot not.

    Parameters:
    ----------
    toekn: str 
        token pass by the user in header.

    Return:
    ------
        return True if the toekn is valid otherwise return False.

    """
    load_dotenv()
    my_key = os.getenv('api-key')
    if token != my_key:
        return True
    return False

def token_required(func: Any)-> Any:
    """
    method token required will perform the authentication based on taoken.

    Parameters:
    ----------
    func: Any
        arguement ass to the function from request header.

    Return:
    ------
        return the the response of the token authentication.

    """
    @wraps(func)
    def decorated(*args, **kwargs):
        token = None
        if 'api-key' in request.headers:
            token = request.headers['api-key']
        if not token:
            result = make_response(json.dumps(
            {'message'  : 'Token Missing',
            'category' : 'Authorization error',}),
            401)
            return result
        if authorize(token):
            result = make_response(
            json.dumps(
            {'message'  : 'UnAuthorized',
            'category' : 'Authorization error',}),
            401)
            return result
        return func(*args, **kwargs)
    return decorated

def validate_request(request_api: Any)-> bool:
    """
    method will take a json request and perform all validations if the any error 
    found then return error response with status code if data is correct then 
    return data in a list.

    Parameters:
    ----------
    request_api: Request
        contain the request data in file format.

    Return:
    ------
    bool
        return True or False.

    """

    if "data_file" in request_api.files:
        return True
    if "data_file" not in request_api.files:
        return False

def get_data(request_api: Any)-> str:
    """
    method will take request and get data from request then return thhe data.

    Parameters:
    ----------
    request_api: Request
        contain the request data in file format.

    Return:
    ------
    image_file: str
        return the data file as string.

    """
    data_file = request_api.files["data_file"]
    return data_file

def save_file(request_api: Any)-> str:
    """
    method will take request and save data from request in specified folder.

    Parameters:
    ----------
    request_api: Request
        contain the request data in file format.

    Return:
    ------
    None

    """
    data_f = request_api.files["data_file"]
    data_f.save(os.path.join(upload_folder, data_f.filename))

def make_bad_params_value_response()-> Response:
    """
    method will make a error response a return it back.

    Parameters:
    ----------
    None

    Return:
    ------
    Response
        return a response message.

    """
    result = make_response(json.dumps(
        {'message'  : 'Empty data_file key',
        'category' : 'Params error',}),
        400)
    return result

def make_bad_params_key_response()-> Response:
    """
    method will make a error response a return it back.

    Parameters:
    ----------
    None

    Return:
    ------
    Response
        return a response message.

    """
    result = make_response(json.dumps(
        {'message'  : 'No image_file found',
        'category' : 'Params error',}),
        400)
    return result

def make_invalid_extension_response()-> Response:
    """
    method will make a error response a return it back.

    Parameters:
    ----------
    None

    Return:
    ------
    Response
        return a response message.

    """
    result = make_response(json.dumps(
        {'message'  : 'Invalid Extension',
        'category' : 'Params error',}),
        400)
    return result

def validate_extension(image: Any)-> bool:
    """
    method will take image and check its extension is .jpg, .jpeg, .png.

    Parameters
    ----------
    image: Any
        image recieved in API request.

    Return
    ------
    bool
        return the true or false image is has valid extension or not.

    """
    images_extensions = ['jpg', 'jpeg', 'png']
    image_ex = imghdr.what(image)
    if image_ex in images_extensions:
        return True
    return False

def validate_video_extension(request_api: Any)-> bool:
    """
    method will take image and check its extension is .mp4, .avi, .FLV.

    Parameters
    ----------
    video: Any
        api request send by user.

    Return
    ------
    bool
        return the true or false video is has valid extension or not.

    """
    video_f = request_api.files["data_file"]
    video_list = video_f.filename.split(".")
    video_extension = video_list[len(video_list)-1]
    video_extensions = ['mp4', 'avi', 'FLV']
    if video_extension in video_extensions:
        return True
    return False

def store_cloud_file(file_path: str)-> str:
    """
    methos will take the file path as input and sotre that file on cloud.

    Parameters
    ----------
    file_path: str
        file path which will be uploaded on gofilecloud.
    
    Return
    ------
        return the file cloud storage url.
    """
    cur_server = go.getServer()
    print(cur_server)
    url = go.uploadFile(file_path)
    print("Download Link: ", url["downloadPage"])
    return url["downloadPage"]
def save_video_file(request_api: Any)-> str:
    """
    method will take request and save file from request in specified folder.

    Parameters:
    ----------
    request_api: Request
        contain the request data in file format.

    Return:
    ------
    save_file_path: str
        file path save on our local sever.
    """
    video_f = request_api.files["data_file"]
    time_stamp_name = str(datetime.datetime.now().timestamp()).replace(".", "")
    video_list = video_f.filename.split(".")
    video_extension = video_list[len(video_list)-1]
    save_file_path = f"{upload_folder}/{time_stamp_name}.{video_extension}"
    file_name = f"{time_stamp_name}.{video_extension}"
    video_f.save(save_file_path)
    return save_file_path, file_name, time_stamp_name

@app.route('/detect_image', methods = ['POST'])
@token_required
@cross_origin()
def detect_object():
    """
    method will take the image as input and return the objects detected 
    in image in a json format.

    Parameters:
    ----------
    None

    Return:
    ------
    str
        return the prediction of the model.

    """
    try:
        if validate_request(request):
            image_file = get_data(request)
            if validate_extension(image_file):
                save_file(request)
                boxes_data = detect_image_objects.detect_objects_on_image(Image.
                                                    open(image_file.
                                                        stream))
                return Response(
                    json.dumps(boxes_data),
                    mimetype = 'application/json'
                    )
            return make_invalid_extension_response()
        return make_bad_params_key_response()
    except Exception as exception:
        return exception

@app.route('/track_video', methods = ['POST'])
@token_required
@cross_origin()
def track_object():
    """
    method will take the video data and return the video with track objects and thier
    counting video with subtitles.

    Parameters:
    ----------
    None

    Return:
    ------
    str
        return the tracked video prediction of the model.

    """
    try:
        if validate_request(request):
            if validate_video_extension(request):
                video_path, file_name, time_stamp = save_video_file(request)
                os.system("python track_video.py --source "+video_path+ " --yolo-weights weights/yolov5n.pt --save-txt --save-vid --count  --time-stamp "+time_stamp)
                print(r"results/output_"+file_name)
                file_url = store_cloud_file(r"results/output_"+file_name)
                output_dict = {}
                output = {
                'video_file' : file_url
                }
                output_dict["output"] = output
                return Response(
                    json.dumps(output_dict),
                    mimetype = 'application/json'
                    )
            return make_invalid_extension_response()
        return make_bad_params_key_response()
    except Exception as exception:
        return exception

if __name__=='__main__':
    app.run(debug = True)