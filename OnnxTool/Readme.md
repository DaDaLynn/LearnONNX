### getOnnxInfo.py  
```
python getOnnxInfo.py
请输入onnx模型路径:
>>0:退出,1:node数量,2:node类型,3:node名,4:model输入,5:model输出,6:model名,7:将整个onnx模型保存为txt
```

### getAllNodes  
```
python getAllNodes.py
>>请输入onnx模型路径:你的onnx模型所在路径,如/xxx/xxx/xxx.onnx
>>请输入节点onnx保存路径(默认路径:../OnnxModel/OnnxNodes/,default选择默认路径):这里还是自己设置比较好，默认路径写的有点问题
>>请输入节点onnx保存形式(onnx或txt):onnx为模型文件，txt的话可读性好
```
### getSingleNode.py  
```
python getSingleNode.py
>>请输入onnx模型路径:
>>请输入节点名(字符串)或节点index(整数):若模型onnx的node name不是空值，可以根据node name来获取相应节点，若没有node name，也可以使用下标获取相应node
>>请输入节点onnx保存路径(默认路径:/OnnxTool/OnnxModel/,default选择默认路径):与getAllNodes一样，推荐自己写一下路径
>>请输入节点onnx保存形式(onnx或txt):与getAllNodes一样
```
