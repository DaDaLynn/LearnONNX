import onnx
from onnx import helper


#加载模型
def loadOnnxModel(path):
    model = onnx.load(path)
    return model

#获取节点和节点的输入输出名列表，一般节点的将来自于上一层的输出放在列表前面，超参数放在列表后面
def getNodeAndIOname(nodename,model):
    print("nodename",nodename)
    if nodename.isdigit():
        index = int(nodename)
        Node = model.graph.node[index]
        input_name = model.graph.node[index].input
        output_name = model.graph.node[index].output
    else:
        for i in range(len(model.graph.node)):
            if model.graph.node[i].name == nodename:
                Node = model.graph.node[i]
                input_name = model.graph.node[i].input
                output_name = model.graph.node[i].output
    return Node,input_name,output_name

#获取对应输入信息
def getInputTensorValueInfo(input_name,model):
    in_tvi = []
    for name in input_name:
        for params_input in model.graph.input:
            if params_input.name == name:
               in_tvi.append(params_input)
        for inner_output in model.graph.value_info:
            if inner_output.name == name:
                in_tvi.append(inner_output)
    return in_tvi

#获取对应输出信息
def getOutputTensorValueInfo(output_name,model):
    out_tvi = []
    for name in output_name:
        out_tvi = [inner_output for inner_output in model.graph.value_info if inner_output.name == name]
        if name == model.graph.output[0].name:
            out_tvi.append(model.graph.output[0])
    return out_tvi

#获取对应超参数值
def getInitTensorValue(input_name,model):
    init_t = []
    for name in input_name:
        init_t = [init for init in model.graph.initializer if init.name == name]
    return init_t

#构建单个节点onnx模型
def createSingelOnnxModel(ModelPath, nodename, SaveType="onnx", SavePath="/home/hanamaru/project/OnnxTool/OnnxModel/"):
    model = loadOnnxModel(ModelPath)
    getSingleOnnxNode(model,nodename,SaveType,SavePath)


def getSingleOnnxNode(model,nodename,SaveType,SavePath):
    Node, input_name, output_name = getNodeAndIOname(nodename, model)
    in_tvi = getInputTensorValueInfo(input_name, model)
    out_tvi = getOutputTensorValueInfo(output_name, model)
    init_t = getInitTensorValue(input_name, model)

    graph_def = helper.make_graph(
        [Node],
        nodename,
        inputs=in_tvi,  # 输入
        outputs=out_tvi,  # 输出
        initializer=init_t,  # initalizer
    )
    model_def = helper.make_model(graph_def, producer_name='onnx-example')
    print(model.graph.name + "-", nodename + "onnx模型生成成功！保存中...")
    if SaveType == "txt":
        with open(SavePath+model.graph.name+"-"+nodename+".txt","w") as f:
            print(model_def,file=f)
            print("已保存为"+model.graph.name+"-"+nodename+".txt文件至"+SavePath)
    if SaveType == "onnx":
        onnx.save_model(model_def,SavePath+model.graph.name+"-"+nodename+".onnx")
        print("已保存为"+model.graph.name+"-"+nodename+".onnx文件至" + SavePath)


def createAllOnnxNodes(ModelPath,SaveType="onnx",SavePath="/home/hanamaru/project/OnnxTool/OnnxModel/OnnxNodes/"):
    model = loadOnnxModel(ModelPath)
    for i in range(len(model.graph.node)):
        getSingleOnnxNode(model,str(i),SaveType,SavePath)

