from OnnxLib.OnnxInfo import *
def main():
    model_path = input("请输入onnx模型路径:")
    model = loadOnnxModel(model_path)
    FLAG = True
    while FLAG:
        operate = input("0:退出,1:node数量,2:node类型,3:node名,4:model输入,5:model输出,6:model名")
        if operate == "0":
            FLAG = False
        elif operate == "1":
            print("node数量：",getNodeNum(model))
        elif operate == "2":
            print("node类型：",getNodetype(model))
        elif operate == "3":
            print("node名：",getNodeNameList(model))
        elif operate == "4":
            print("model输入：",getModelInputInfo(model))
        elif operate == "5":
            print("model输出：",getModelOutputInfo(model))
        elif operate == "6":
            print("model名：",getModelName(model))
        else:
            print("没有这个操作")

if __name__ == '__main__':
    main()
