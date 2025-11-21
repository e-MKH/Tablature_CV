from roboflow import Roboflow

rf = Roboflow(api_key="dQjY0kFBJvmuLawGVH9I")
project = rf.workspace("capstone-ahtph").project("bass-guitar-qfwfd")
version = project.version(1)
dataset = version.download("yolov11")
