from OnnxLib.NodeTemp import *

def main():
    model_path = input("请输入onnx模型路径:")
    save_path = input("请输入节点onnx保存路径(默认路径:../OnnxModel/OnnxNodes/,default选择默认路径):")
    save_type = input("请输入节点onnx保存形式(onnx或txt):")
    if save_path == "default":
        createAllOnnxNodes(model_path,save_type)
    else:
        createAllOnnxNodes(model_path,save_type,save_path)

if __name__ == '__main__':
    main()
