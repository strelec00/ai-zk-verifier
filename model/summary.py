import onnx
from onnx import helper

model = onnx.load("cat_dog_classifier.onnx")
print(onnx.helper.printable_graph(model.graph))