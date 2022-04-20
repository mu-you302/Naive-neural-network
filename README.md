# Naive-neural-network
Implementing Neural Networks with Numpy.


## 用numpy实现简单的神经网络
---
模型的任务是实现sklearn中digits手写数字分类。

第一个版本 *NN* 是分散式的，不利于扩展，第二个版本 *LtoL* 将层定义为对象，实现了模块化，想构建更多层的神经网络更容易。

将正向传播和反向传播的过程封装在层class的内部，只需要以正确的顺序连接各层，在按照正确的顺序（或者逆序）调用各层，`逐层传播` 的实现就变得很容易。

更详细的讲解参见《深度学习入门：基于Python的理论与实现》
