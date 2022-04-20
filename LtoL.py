from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split


class Sigmoid:
    """
    隐层使用的simoid函数"""

    def __init__(self):
        self.y = None  # 保存sigmoid作用之后的输出

    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, dout):
        return dout * (self.y - self.y**2)


class Relu:
    """
    ReLU函数，但是在本网络中没有用到"""

    def __init__(self):
        self.mask = None  # 保存输入中<=0的部分

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        """
        ReLU的导数非0即1，为0的部分会被截去，为1的部分保持不变"""
        dout[self.mask] = 0
        return dout


class Softmax:
    """
    输出层使用的softmax函数"""

    def __init__(self) -> None:
        self.y = None

    def forward(self, x):
        C = np.max(x)
        s = np.exp(x - C)
        self.y = s / np.sum(s)
        return self.y

    def backward(self, dout):
        """
        这里仅仅是i=j时的softmax求导结果"""
        return dout * (self.y - self.y**2)


class Affine:
    """
    神经网络中正向传播中矩阵运算称为仿射变换（Affine）"""

    def __init__(self, W, b) -> None:
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x.reshape(1, -1)  # 将输入调形为行向量
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        """
        暂存dW和db并输出dx"""
        self.dW = np.dot(self.x.T, dout.reshape(1, -1))
        # self.db = np.sum(dout, axis=0)  # 累积BP算法的写法
        self.db = dout
        return np.dot(dout, self.W.T)


class TriLayerNet:
    """
    三层（输入层、隐层、输出层）的简单神经网络"""

    def __init__(
        self, input_size, hidden_size, output_size, weight_init_std=0.01, alpha=0.1
    ) -> None:
        self.alpha = alpha  # 学习率
        self.losses = []  # 记录每轮训练中的loss

        self.params = {}  # 神经网络的参数，即各层中的W和b
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

        self.layers = {}  # 神经网络中的各层，包含使用的非线性函数
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Sigmoid1"] = Sigmoid()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
        self.layers["Softmax"] = Softmax()

    def forward(self, x):
        """
        前向传播产生结果，注意是输出层的10个预测结果"""
        for layer in self.layers.values():  # 用各层对象进行迭代
            x = layer.forward(x)
        return x

    def predict(self, x):
        """
        每个样本一个结果"""
        res = []
        for i in range(x.shape[0]):
            out = self.forward(x[i])
            res.append(np.argsort(-out)[0])  # 将out中的内容按照降序排列，取出最大预测值对应的索引
        return res

    def loss(self, y, t):
        """
        计算均方误差MSE，参数t为真实值的onehot编码"""
        return 0.5 * np.sum((y - t) ** 2)

    def backward(self, x, t):
        """
        误差逆传播的过程，x表示数据的特征，t表示label的onehot编码"""

        y = self.forward(x)

        dout = y - t  # 采用MSE时的损失函数的导数

        l = list(self.layers.values())  # 各层的对象

        # 误差逆传播的过程，将逐层的特性体现的淋漓尽致
        for layer in l[::-1]:
            dout = layer.backward(dout)

        self.params["W1"] -= self.alpha * self.layers["Affine1"].dW
        self.params["b1"] -= self.alpha * self.layers["Affine1"].db
        self.params["W2"] -= self.alpha * self.layers["Affine2"].dW
        self.params["b2"] -= self.alpha * self.layers["Affine2"].db

    def fit(self, x, t, epoch):
        for _ in range(epoch):
            l = []
            for i in range(x.shape[0]):
                t_onehot = np.eye(10)[t[i]]  # 生成onehot编码
                l.append(self.loss(self.forward(x[i]), t_onehot))
                self.backward(x[i], t_onehot)
            self.losses.append(np.mean(l))


X, y = load_digits(return_X_y=True)

# 数据标准化，均值为0，方差为1，标准化之后性能有所提升
X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = TriLayerNet(64, 47, 10)

model.fit(X_train, y_train, 17)

y_predict = model.predict(X_test)

# 模型的score
print("The accuracy is:", accuracy_score(y_test, y_predict))
print("The MSE is", mean_squared_error(y_test, y_predict))

# 绘制每轮epoch中的loss情况
plt.plot(list(range(17)), model.losses, "r")
plt.show()
