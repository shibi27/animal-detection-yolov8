!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="wLvpqf00Q3RZ6DYpsKFV")
project = rf.workspace("shibi-p-xmggq").project("animal-detection-using-yolov8-cvuyo")
version = project.version(2)
dataset = version.download("yolov8")
