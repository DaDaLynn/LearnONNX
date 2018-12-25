from OnnxLib.NodeTemp import *

def main():
    model_path = input("请输入onnx模型路径:")
    node_name = input("请输入节点名(字符串)或节点index(整数):")
    save_path = input("请输入节点onnx保存路径(默认路径:/OnnxTool/OnnxModel/,default选择默认路径):")
    save_type = input("请输入节点onnx保存形式(onnx或txt):")
    if save_path == "default":
        createSingelOnnxModel(model_path,node_name,save_type)
    else:
        createSingelOnnxModel(model_path,node_name,save_type,save_path)

if __name__ == '__main__':
    main()
