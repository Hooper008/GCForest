
# GCForest
---
实现了周志华论文中的GCForest算法
### 介绍
gcForest(multi-Grained Cascade forest 多粒度级联森林)是周志华教授最新提出的新的决策树集成方法。这种方法生成一个深度树集成方法（deep forest ensemble method），使用级联结构让gcForest学习。 
gcForest模型把训练分成两个阶段：Multi-Grained Scanning和Cascade Forest。Multi-Grained Scanning生成特征,Cascade Forest经过多个森林多层级联得出预测结果。

### Cascade Forest(级联森林)
![](https://upload-images.jianshu.io/upload_images/2764844-4d87815b69c4e1db.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
1. 级联中的每一级接收到由前一级处理的特征信息，并将该级的处理结果输出给下一级。 
2. 级联的每个级别包括两个随机森林（蓝色字体标出）和两个完全随机树木森林（黑色）。[可以是多个，为了简单这里取了2种森林4个弱分类器] 
3. 每个完全随机的树森林包含1000(超参数)个完全随机树，通过随机选择一个特征在树的每个节点进行分割实现生成，树一直生长，直到每个叶节点只包含相同类的实例或不超过10个实例。 
4. 类似地，每个随机森林也包含1000(超参数)棵树，通过随机选择√d数量(输入特征的数量开方)的特征作为候选，然后选择具有最佳gini值的特征作为分割。(每个森林中的树的数值是一个超参数) 
假设有三个类要预测; 因此，每个森林将输出三维类向量，然后将其连接输入特征以重新表示下一次原始输入。
类别概率向量生成: 
给定一个实例，每个森林会通过计算在相关实例落入的叶节点处的不同类的训练样本的百分比，然后对森林中的所有树计平均值，以生成对类的分布的估计。即每个森林会输出一个类别概率向量。 
为了降低过拟合风险，每个森林产生的类向量由k折交叉验证（k-fold cross validation）产生。 
![](http://upload-images.jianshu.io/upload_images/2764844-c20df99b261a3b90?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

假设有三个类，则四个森林每一个都将产生一个三维的类向量，因此，级联的下一级将接收12 = 3×4个增强特征（augmented feature） 
![这里写图片描述](http://upload-images.jianshu.io/upload_images/2764844-a120649e2bb8fe8a?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## Multi-Grained Scanning(多粒度扫描)

用多粒度扫描流程来增强级联森林,使用滑动窗口扫描的生成实例，输入森林后结果合并，生成新的特征。 
![](http://upload-images.jianshu.io/upload_images/2764844-49eaf6d4d8ca9d9c?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

假设有400个原始特征，并且使用100个特征的窗口大小。对于序列数据，将通过滑动一个特征的窗口来生成100维的特征向量；总共产生301个实例100维的特征向量。从相同大小的窗口提取的实例将用于训练完全随机树森林和随机森林，然后经过训练后的森林生成类向量并将301实例类别概率维度连接为转换后的特征。 
维度变化：1个实例400维->301个实例100维->2棵森林301个实例3维->1806维(2x301x3)

## 整体流程

![](http://upload-images.jianshu.io/upload_images/2764844-eb7ee975ad0aa06e?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

阶段1： 
1\. 利用滑动窗口切分成多实例特征向量，经过森林变换输出类别概率向量。 
2\. 合并类别概率向量生成新的特征。 
阶段2： 
3\. 输入特征经过森林输出类别概率向量，连接原始输入作为下一层输出。 
4\. 经过多个级联森林，输出最终的类别概率向量。 
5\. 对多个森林输出的类别概率向量求类别的均值概率向量，取最大的类别概率为预测结果。

## 优点

1.  性能较之深度神经网络有很强的竞争力。
2.  gcForest较深度神经网络容易训练得多
3.  gcForest具有少得多的超参数，并且对参数设置不太敏感，在几乎完全一样的超参数设置下，在处理不同领域的不同数据时，也能达到极佳的性能，即对于超参数设定性能鲁棒性高。
4.  训练过程效率高且可扩展，适用于并行的部署，其效率高的优势就更为明显。
5.  gcForest在仅有小规模训练数据的情况下也表现优异。


