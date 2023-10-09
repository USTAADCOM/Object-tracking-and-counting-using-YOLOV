"""
Module contain the methods which predict the sales for the given set of data
"""
from ultralytics import YOLO

def load_model():
    """
    method will load the model and return loaded model.

    Parameters:
    ----------
    None

    Return:
    ------
    model: Any
        return the model object loaded from best.pt file.

    """
    model = YOLO("best.pt")
    return model

def detect_objects_on_image(images):
    """
    method will take the data and return the SSPL prediction,

    Parameters:
    ----------
    None

    Return:
    ------
    str
        return the prediction of the model as dictinary.

    """
    model = load_model()
    results = model.predict(images)
    result = results[0]
    count = 0
    output_dict = {}
    for box in result.boxes:
        x_one, y_one, x_two, y_two = [
          round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output = {'shape_box' : [x_one, y_one, x_two, y_two],
                       'object' : result.names[class_id],
                       'confidence' : prob}
        count+=1
        output_dict["object" + str(count)] = output
    return output_dict
