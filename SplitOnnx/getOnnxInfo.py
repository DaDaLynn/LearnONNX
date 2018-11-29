import onnx
import sys,getopt
#获取节点数量
def getNodeNum(model):
    return len(model.graph.node)
#获取节点类型
def getNodetype(model):
    op_name = []
    for i in range(len(model.graph.node)):
        if model.graph.node[i].op_type not in op_name:
            op_name.append(model.graph.node[i].op_type)
    return op_name
#获取节点名列表
def getNodeNameList(model):
    NodeNameList = []
    for i in range(len(model.graph.node)):
        NodeNameList.append(model.graph.node[i].name)
    return NodeNameList
#获取模型的输入信息
def getModelInputInfo(model):
    return model.graph.input[0]
#获取模型的输出信息
def getModelOutputInfo(model):
    return model.graph.output[0]


def getInfo(path):
    model = onnx.load(path)
    print("节点数量：", getNodeNum(model))
    print("节点类型有：",getNodetype(model))
    print("节点名列表：",getNodeNameList(model))
    print("模型输入：",getModelInputInfo(model))
    print("模型输出：",getModelOutputInfo(model))


def usage():
    print("""
    python getOnnxInfo.py modelpath 
    """)


def main():
    if len(sys.argv) == 1:
        usage()
        sys.exit()
    model_path = sys.argv[1]
    getInfo(model_path)

if __name__ == '__main__':
    main()
