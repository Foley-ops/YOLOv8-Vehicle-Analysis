from roboflow import Roboflow

rf = Roboflow(api_key="7eZKA9SZR2FCTRX8SHu8")
project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
version = project.version(6)
dataset = version.download("yolov8")
