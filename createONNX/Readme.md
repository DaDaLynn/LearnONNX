## ONNX结构分析
onnx将每一个网络的每一层或者说是每一个算子当作节点**Node**，再由这些**Node**去构建一个**Graph**，相当于是一个网络。最后将**Graph**和这个onnx模型的其他信息结合在一起，生成一个**model**，也就是最终的.onnx的模型。
### onnx.helper----node、graph、model
在构建onnx模型这个过程中，这个文件至关重要。其中**make_node**、**make_graph**、**make_model**是不可或缺的。**make_tensor_value_info**和**make_tensor**是构建graph中所需要用到的。
#### make_node [类型:NodeProto]
make_node(op_type,inputs,outputs,name=None,doc_string=None,**kwargs)
- **op_type**:节点的算子类型 [类型:字符串]
比如Conv、Relu、Add这类，详细可以参考[onnx给出的算子列表](https://github.com/onnx/onnx/blob/f2daca5e9b9315a2034da61c662d2a7ac28a9488/docs/Operators.md)，这个可以自己赋值，但最好与官网对应上，否则其他框架在跑onnx的时候会不知道这是什么。
- **inputs**:存放节点输入的名字 [类型:字符串列表]
每个节点输入的数量根据情况会有不同，比如inputs(2-3)，即输入为2个或3个，可选的输入都会标注(optional)。以Conv为例，必有输入X和权重W，偏置B作为可选。  
- **outputs**:存放节点输出的名字 [类型:字符串列表]
与**inputs**类似，同样需要根据官网给出的输出个数来设置，大多数情况是一个输出，我暂且还没碰到多输出情况。
- **name**:节点名，可有可无，不要和op_type搞混了
- **doc_string**:描述文档的字符串，这个默认为None [类型:字符串]
- ****kwargs**:存放节点的属性attributes [类型:任意]
这个****kwargs**可以是字典形式输入，也可以拆开分别赋值(类型任意)，反正不管是什么最后这个node都给你转换成NodeProto的形式。在用IDE的时候，你可以进到一个onnx_ml_pb2.py的文件中，你可以看到诸如AttributeType、DataType、AttributeProto、ValueInfoProto、NodeProto这些描述符号。onnx_ml_pb2.py是由protoc buffer编译器通过onnx-ml.proto生成的。
Attributes在官网也被明确的给出了，一般被标注(default:xxxxx)的可以根据自己的需求不设置，没有标注default的属性则一定需要设置。
以Conv举例：
`auto_pad`:VALID,`dilations`:[1,1,1],`group`:1,`kernel_shape`:(7,7),`pads`:[3,3,3,3],`strides`:(2,2)
可以写成：
```
dict = {"kernel_shape": (7, 7),
"group": 1,#default为1，所以可以不写
"strides": (2, 2), 
"auto_pad": "VALID", 
"dilations": [1, 1, 1],
"pads": [3, 3, 3, 3]}#顺序无所谓
node_def = helper.make_node(
        NodeType,  # 节点名
        X_name,  # 输入
        Y_name,  # 输出
        **dict
    )
```
也可以写成：
```
node_def = helper.make_node(
        NodeType,  # 节点名
        X_name,  # 输入
        Y_name,  # 输出
        kernel_shape = (7,7),
        strides = (2,2),
        auto_pad = "VALID",
        dilations = [1,1,1],
        pads = [3,3,3,3], 
    )
```
当然你也可以自己魔改，想放什么进去都可以，不过尽量还是统一符合官网要求比较好~


#### make_graph [类型:GraphProto]
make_graph(nodes,name,inputs,outputs,initializer=None,doc_string=None,value_info=[])
- **nodes**:用make_node生成的节点列表 [类型:NodeProto列表]
比如[node1,node2,node3,...]这种的
- **name**:graph的名字 [类型:字符串]
- **inputs**:存放graph的输入数据信息 [类型:ValueInfoProto列表]
输入数据的信息以ValueInfoProto的形式存储，会用到**make_tensor_value_info**，来将输入数据的名字、数据类型、形状(维度)给记录下来。
- **outputs**:存放graph的输出数据信息 [类型:ValueInfoProto列表]
与**inputs**相同。
- **initializer**:存放超参数 [类型:TensorProto列表]
比如Conv的权重W、偏置B，BatchNormalization的scale、B、mean、var。这些参数数据都是通过**make_tensor**来转换成TensorProto形式。
- **doc_string**:描述文档的字符串，这个默认为None [类型:字符串]
- **value_info**:存放中间层产生的输出数据的信息 [类型:ValueInfoProto列表]

**注意！**inputs、outputs、value_info都是ValueInfoProto列表形式，那么它们各自存放什么东西呢？
对于一个多层网络而言，其中间层的输入有来自上一层的输出，也有来自外界的超参数和数据，为了区分，onnx中将来自外界的超参数信息和输入数据信息统一放在inputs里，而value_info里存放的是来自经过前向计算得到的中间层输出数据的信息。注意，是信息，不是具体数据值。outputs只存放整个网络的输出信息，也就是说，只有一个。

**第二个需要注意的是：**initializer作为存放超参数具体数值的TensorProto列表，其中每个TensorProto总会有与其对应的ValueInfoProto存在，对应关系通过name来联系。比如inputs里放了一个Conv1的权重参数信息，名字为"Conv1_W"那么对应的initializer里会有个名字与其相同的TensorProto来存储这个权重参数的具体数值。

**第三个需要注意的是：**对于一个网络而言如何能体现其网络结构呢？即节点与节点之间的关联。
在构建每一个node时就需要注意，当前node的输入来自于哪一个node的输出，名字要匹配上，才能将node间联系体现出来。

#### make_model
make_model(graph, **kwargs)
- **graph**:用make_graph生成的GraphProto
- ****kwargs**:构建ModelProto中的opset_import，这个还没弄太清楚，不过不影响生成模型

这个函数中会先实例化一个ModelProto----model，其中会对它的ir_version(现在默认是3)、graph(就是把传入的graph复制进model.graph)、opset_import做处理。具体可以看helper里的make_model这个函数。我们只要知道这是个最后把graph和模型其他信息组合在一起构建出一个完整的onnx model的函数就可以了。

### onnx.helper----tensor、tensor value info、attribute
#### make_tensor [类型:TensorProto]
make_tensor(name,data_type,dims,vals,raw=False)
-  **name**:数据名字，要与该数据的信息tensor value info中名字对应 [类型:字符串]
-  **data_type**:数据类型 [类型:TensorProto.DataType] 如TensorProto.FLOAT、TensorProto.UINT8、TensorProto.FLOAT16等
-  **dims**:数据维度 [类型:int列表/元组]
-  **vals**:数据值，好像要可迭代的 [类型:任意]
-  **raw**:选择是否用二进制编码 [类型:bool]
raw为False的时候，就会用相应的TensorProto来存储基于data_type的值，若raw为True，则是用二进制编码来存储数据。
**注：**我发现cntk官方转onnx用的是raw为False的方式，而pytorch官方转onnx用的是raw为True的方式。

#### make_tensor_value_info [类型:ValueInfoProto]
make_tensor_value_info(name,elem_type,shape,doc_string="",shape_denotation=None)
-  **name**:数据信息名字 [类型:字符串]
-  **elem_type**:数据类型 [类型:TensorProto.DataType]
-  **shape**:数据维度(形状) [类型:int列表/元组]
-  **doc_string**:描述文档的字符串，这个默认为None [类型:字符串]
-  **shape_denotation**:这个没太看懂，可能是对shape的描述 [类型:字符串列表]
根据数据类型和形状创建一个ValueInfoProto。

#### make_attribute [类型:AttributeProto]
make_attribute(key,value,doc_string=None)
- **key**:键值 [类型:字符串]
- **value**:数值 [类型:任意]
- **doc_string**:描述文档的字符串，这个默认为None [类型:字符串]
根据数值类型来创建一个AttributeProto，这个函数用在了make_node里，用于将make_node传入的**kwargs转为AttributeProto形式。


**构建一个简单的onnx模型，实质上，只要构建好每一个node，然后将它们和输入输出超参数一起塞到graph，最后转成model就可以了。**

写了一个base，在构建onnx的时候可以直接调用createOnnxNode、createOnnxModel来构建一个onnx模型，可以选择把onnx保存为txt格式，很大就是了。具体流程后续补上。
[代码](https://github.com/htshinichi/LearnONNX/blob/master/createONNX/base.py)
