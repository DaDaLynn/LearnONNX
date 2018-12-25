#参考于https://github.com/onnx/onnx/blob/6bedd27b0307c9295039bd847895a27275160a98/onnx/examples/np_array_tensorproto.ipynb
from onnx import numpy_helper
import onnx
# model_path = "/home/hanamaru/project/OnnxTool/OnnxModel/FromCaffe/resnet50.onnx"
##导入你的onnx模型
model_path = "/home/hanamaru/project/OnnxTool/OnnxModel/OnnxNodes/torch-jit-export-0.onnx"
model = onnx.load_model(model_path)

##遍历这个模型中所有initializer，并转换为数组形式
for init in model.graph.initializer:
    print(init)
    data = numpy_helper.to_array(init)
    print(data)
