### Tensor2Array.py  
用于将被编码为二进制的数值解码

### getOnnxInfo.py  
```shell
python getOnnxInfo.py
>>请输入onnx模型路径:你的onnx模型所在路径,如/xxx/xxx/xxx.onnx
>>0:退出,1:node数量,2:node类型,3:node名,4:model输入,5:model输出,6:model名,7:将整个onnx模型保存为txt
```

### getAllNodes  
用于将一个完整的ONNX模型拆分出所有单个node的onnx模型。
```shell
python getAllNodes.py
>>请输入onnx模型路径:你的onnx模型所在路径,如/xxx/xxx/xxx.onnx
>>请输入节点onnx保存路径(默认路径:../OnnxModel/OnnxNodes/,default选择默认路径):这里还是自己设置比较好，默认路径写的有点问题
>>请输入节点onnx保存形式(onnx或txt):onnx为模型文件，txt的话可读性好
```
这里建议自己查看一下onnx模型是否有value_info，如果没有，则拆分出来的单个节点的onnx模型是没有outpout的。我发现pytorch转出来的onnx模型没有node name，也没有value_info，make_tensor的时候也是选择了用raw将数据进行了二进制编码。虽然这些确实不会影响onnx的推理，但在做拆分或者验证比对的时候确实有一些难受(╯﹏╰)。  

```python
#查看方法
import onnx
model = onnx.load(your onnx model path)
print(model.graph.value_info)
```
### getSingleNode.py  
用于拆分出单个node的onnx模型
```shell
python getSingleNode.py
>>请输入onnx模型路径:你的onnx模型所在路径,如/xxx/xxx/xxx.onnx
>>请输入节点名(字符串)或节点index(整数):若模型onnx的node name不是空值，可以根据node name来获取相应节点，若没有node name，也可以使用下标获取相应node,node name可以自己查看，也可以通过getOnnxInfo来看
>>请输入节点onnx保存路径(默认路径:/OnnxTool/OnnxModel/,default选择默认路径):与getAllNodes一样，推荐自己写一下路径
>>请输入节点onnx保存形式(onnx或txt):与getAllNodes一样
```
