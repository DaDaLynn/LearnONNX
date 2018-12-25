import onnx

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

def getModelName(model):
    return model.graph.name

def saveOnnx2Txt(model,save_path):
    with open(save_path+model.graph.name+".txt","w") as f:
        print(model,file=f)

#加载模型
def loadOnnxModel(path):
    model = onnx.load(path)
    return model

