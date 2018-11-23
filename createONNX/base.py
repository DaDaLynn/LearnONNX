import onnx
from onnx import helper
from onnx import TensorProto

#将数据转为list形式
def __getDataList(data):
    data2list = data.reshape(1,-1).tolist()[0]
    return data2list

#将数据转为Tensor形式、并建立数据信息
def __makeTensorAndGetInfo(tensor_name,data,shape):
    #先将data转为list形式
    data_list = __getDataList(data)
    # 将输入数据、输出数据、参数转为tensor形式
    data_tensor = helper.make_tensor(name=tensor_name,dims=shape,data_type=TensorProto.FLOAT,vals=data_list)
    # 建立输入的数值信息
    data_tensor_info = helper.make_tensor_value_info(tensor_name,TensorProto.FLOAT,shape)
    return [data_tensor,data_tensor_info]

#创建输入输出数据的信息
def __createIOInfo(name,data,shape):
    name_list = []
    tensor_list = []
    info_list = []
    if len(shape)==1:
        x = __makeTensorAndGetInfo(name[0],data[0],shape[0])
        name_list.append(name[0])
        tensor_list.append(x[0])
        info_list.append(x[1])
    else:
        for i in range(len(shape)):
            x = __makeTensorAndGetInfo(name[i], data[i], shape[i])
            name_list.append(name[i])
            tensor_list.append(x[0])
            info_list.append(x[1])
    return name_list,tensor_list,info_list

#创建参数数据的信息
def __createParamInfo(NodeName,name,data,shape):
    name_list = []
    tensor_list = []
    info_list = []
    for i in range(len(name)):
        x = __makeTensorAndGetInfo(NodeName+"_"+name[i],data[i],shape[i])
        name_list.append(NodeName+"_"+name[i])
        tensor_list.append(x[0])
        info_list.append(x[1])
    return name_list,tensor_list,info_list


# 创建Onnx节点
def createOnnxNode(NodeType, NodeName, input_data, input_shape,input_name, output_data, output_shape,output_name, dict, haveInput=False,haveOutput=True, param_data=[], param_shape=[], param_name=[]):
    X_name, X_tensor, X_info = __createIOInfo(input_name, input_data, input_shape)
    Y_name, Y_tensor, Y_info = __createIOInfo(output_name, output_data, output_shape)
    Info = [X_info, Y_info]

    # 若是参数列表长度不为0，则将参数名和信息放进输入信息中，将参数值放入Init（即Info[2]）中
    if len(param_name) != 0:
        P_name, P_tensor, P_info = __createParamInfo(NodeName, param_name, param_data, param_shape)
        X_name.extend(P_name)
        X_info.extend(P_info)
        Info.append(P_tensor)
    # 选择是否将输入数据放入结点
    if haveInput:
        for i in range(len(X_tensor)):
            dict["input" + str(i + 1)] = X_tensor[i]

    # 选择是否将输出数据放入结点
    if haveOutput:
        for j in range(len(Y_tensor)):
            dict["output" + str(j + 1)] = Y_tensor[j]

    node_def = helper.make_node(
        NodeType,  # 节点名
        X_name,  # 输入
        Y_name,  # 输出
        **dict
    )

    return [node_def], Info


# 创建Onnx模型
def createOnnxModel(OP_name, node_def, in_Info, out_Info, Path, FileName, savetxt=True, init_Info=None):
    if init_Info is None:
        # 创建模型的图信息
        graph_def = helper.make_graph(

            node_def,
            OP_name,
            in_Info,  # 输入.
            out_Info,  # 输出
        )
    else:
        # 创建模型的图信息
        graph_def = helper.make_graph(
            node_def,
            OP_name,
            in_Info,  # 输入
            out_Info,  # 输出
            init_Info,  # initalizer
        )
    # 创建模型
    model_def = helper.make_model(graph_def, producer_name='onnx-example')
    # print(FileName + " ops:\n", model_def)
    onnx.save_model(model_def, Path + FileName + ".onnx")

    # 选择是否将onnx打印后保存为txt
    if savetxt:
        ONNX2TXT(Path, FileName)
        f = open(Path + FileName + ".txt", "w")
        print(model_def, file=f)
        f.close()


def ONNX2TXT(Path, FileName):
    model = onnx.load(Path + FileName + ".onnx")
    f = open(Path + FileName + ".txt", "w")
    print(model, file=f)
    f.close()
