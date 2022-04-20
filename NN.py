import numpy as np
from sklearn import preprocessing

from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, mean_squared_error

import matplotlib.pyplot as plt

# 定义sigmoid和ReLU两种激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


# sigmoid导数，dy/dx，这里传入输出y的值就可以得到导数值
def sigmoid_derivative(y):
    return y - y**2


def relu_derivative(x):
    Z = x.copy()  # 从x拷贝一个副本，否则会修改原x中的内容
    Z[Z < 0] = 0
    Z[Z > 0] = 1
    return Z


# 分类问题输出层的结果需要经过softmax函数的处理，使得y在[0,1]之间，并且所有y之和为1
def softmax(y):
    C = np.max(y)
    a = np.exp(y - C)
    return a / np.sum(a)


# softmax函数的导数，在此神经网络中属于i=j的求导情况，结果与sigmoid类似
def softmax_derivative(y):
    return y - y**2


# 输入层神经元的个数应该设置成输入的feature的数量，这里手写数字数据集shape为(1797, 64)

# 在这个简单的神经网络中只设计一层隐层，并取隐层神经元的个数为50个
# 隐层神经元个数太少容易导致欠拟合，太多又容易过拟合并增加训练开销

# 回归任务中输出层神经元只要一个就够了，分类任务中输出层神经元个数应该设置成类别数量
class NN:
    def __init__(
        self, inputnum, hiddennum, outputnum, alpha=0.1, weight_init_std=0.01
    ) -> None:
        # 分别定义各层神经元的数量
        self.input_num = inputnum
        self.hidden_num = hiddennum
        self.output_num = outputnum
        # 两层的偏置
        self.b1 = np.zeros(self.hidden_num)
        self.b2 = np.zeros(self.output_num)
        self.alpha = alpha  # 学习率
        # 产生随机的权重矩阵，这里引入参数weight_init_std将权重初始化为更小的值，能提高预测精度
        # 初始化参数很小，如果学习率也设置的很小如0.01，那么模型的收敛速度会很慢
        self.V = weight_init_std * np.random.randn(inputnum, hiddennum)
        self.W = weight_init_std * np.random.randn(hiddennum, outputnum)
        self.losses = []

    def fit(self, X, y, epoch):
        for n in range(epoch):
            loss = []
            for i in range(X.shape[0]):
                loss.append(self.loss(self.forward(X[i]), y[i]))
                self.BP_pro(X[i], y[i])
            self.losses.append(np.mean(loss))

    # 数据前向传播产生输出
    def forward(self, X):
        a1 = X.dot(self.V) + self.b1
        z1 = sigmoid(a1)
        a2 = z1.dot(self.W) + self.b2
        return softmax(a2)

    # 采用均方误差计算损失函数
    def loss(self, output, y):
        Y = np.array([0] * 10)
        Y[y] = 1
        return 0.5 * np.sum((output - Y) ** 2)

    # BP算法西瓜书版，但是其实这种算法并不利于实现，特别是前面层的参数以及偏置的更新
    def BP(self, X, y):
        a1 = X.dot(self.V)
        z1 = sigmoid(a1)
        a2 = z1.dot(self.W)
        z2 = softmax(a2)

        y_real = np.array([0] * 10)
        y_real[y] = 1

        G = ((z2 - y_real) * softmax_derivative(z2)).reshape(1, -1)
        delta_W = z1.reshape(-1, 1) @ G

        # 输入层与隐层之间参数的更新涉及到矩阵的求导，先用for循环写出来
        E = np.zeros((1, self.hidden_num))
        for i in range(self.hidden_num):
            E[0, i] = np.sum(G[0, :] * self.W[i, :]) * sigmoid_derivative(z1[i])
        delta_V = X.reshape(-1, 1) @ E

        self.W -= self.alpha * delta_W
        self.V -= self.alpha * delta_V

    # BP算法改进版，加入了偏置，也更方便计算梯度在不同层之间的传递
    def BP_pro(self, X, y):
        a1 = np.dot(X, self.V) + self.b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, self.W) + self.b2
        z2 = softmax(a2)

        y0 = np.array([0] * 10)
        y0[y] = 1

        # 特别注意这里应该用预测值减真实值，否则就成了梯度上升了，误差不减反增
        dout = z2 - y0

        da2 = dout * softmax_derivative(z2)

        dW = np.dot(z1.reshape(-1, 1), da2.reshape(1, -1))

        dz1 = np.dot(da2, self.W.T)  # 计算出误差对于z1的导数，这样后面的过程就与前一部分类似

        da1 = dz1 * sigmoid_derivative(z1)

        dV = np.dot(X.reshape(-1, 1), da1.reshape(1, -1))

        self.W -= self.alpha * dW
        self.b2 -= self.alpha * da2
        self.V -= self.alpha * dV
        self.b1 -= self.alpha * da1

    def predict(self, X):
        res = []
        for i in range(X.shape[0]):
            out = self.forward(X[i])
            res.append(np.argsort(-out)[0])  # 将out中的内容按照降序排列
        return res


X, y = load_digits(return_X_y=True)

# 数据标准化，均值为0，方差为1，标准化之后性能有所提升
X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = NN(64, 47, 10)

model.fit(X_train, y_train, epoch=40)  # 模型训练40轮就已经接近收敛了

y_predict = model.predict(X_test)

# 模型的score
print("The accuracy is:", accuracy_score(y_test, y_predict))
print("The MSE is", mean_squared_error(y_test, y_predict))

# 绘制每轮epoch中的loss情况
plt.plot(list(range(40)), model.losses, "r")
plt.show()
