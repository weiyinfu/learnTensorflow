神经网络根本不需要训练
直接随机堆叠若干层，然后只最优化最后一层即可

这就是极限学习机的原理。

中间堆叠的隐层越乱越好，可以使用各种各样的激活函数。

最后一层只是一个简单的softmax最优化，如果数据规模小，可以直接使用数学方法求解。
