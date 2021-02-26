# iris_tensorflow:鸢尾花数据集tensorflow
# 正确率有时高达百分之百
# 这里使用了tensorflow.contrib中的DNNClassifier(深度全连接网络)

import tensorflow as tf
from sklearn import metrics, model_selection, datasets


class MyMonitor(tf.contrib.learn.monitors.BaseMonitor):
    def __init__(self, accuracy, outputStep):
        super(MyMonitor, self).__init__()
        self.accuracy = accuracy
        self.outputStep = outputStep

    def begin(self, max_step=None):
        print("begin", max_steps)

    def end(self, session=None):
        print("end")

    def epoch_begin(self, *arg, **args):
        print("epoch begin", arg, args)

    def epoch_end(self, *arg, **args):
        print("epoch end", arg, args)

    def post_step(self, stepNumber, sess):
        print("post_step", stepNumber)
        if stepNumber % self.outputStep == self.outputStep - 1:
            print(stepNumber, "正确率", self.accuracy())

    def step_begin(self, stepNumber):
        print("step_begin", stepNumber)

    def step_end(self, stepNumber, output):
        print("step_end", stepNumber, output)


def main(_):
    # Load dataset.加载数据的两种方式
    # iris = tf.contrib.learn.datasets.load_dataset('iris')
    iris = datasets.load_iris()
    print(iris)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(
        x_train)
    print(feature_columns)
    print(x_train.dtype, x_train.shape, y_train.dtype, y_train.shape)
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3)

    def test():
        predictions = list(classifier.predict(x_test, as_iterable=True))
        score = metrics.accuracy_score(y_test, predictions)
        return score

    # Fit and predict.
    classifier.fit(x_train, y_train,
                   batch_size=10,
                   steps=200,
                   monitors=[MyMonitor(test, 12)])

    print('Accuracy: {0:f}'.format(test()))


if __name__ == '__main__':
    tf.app.run()
