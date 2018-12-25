from torch.autograd import Variable
import torchvision
import torch
from OnnxModel import ModelPath as MP

#根据onnx官网示例，生成alexnet.onnx模型和resnet50.onnx模型
dummy_input1 = Variable(torch.randn(1, 3, 224, 224))
model_alexnet = torchvision.models.alexnet(pretrained=True)
torch.onnx.export(model_alexnet, dummy_input1, MP.TORCH_PATH+"alexnet.onnx")
dummy_input2 = Variable(torch.randn(10, 3, 224, 224))
model_resnet50 = torchvision.models.resnet50(pretrained=True)
torch.onnx.export(model_resnet50, dummy_input2, MP.TORCH_PATH+"resnet50.onnx", verbose=True)
