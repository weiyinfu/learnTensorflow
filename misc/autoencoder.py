# coding = utf-8
import math

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris


class DataIterator(object):
    def __init__(self, datas, labels=None, batch_size=60, dtype=np.float64):
        if type(datas) == list:
            datas = np.array(datas, dtype=dtype)
        if labels is not None:
            if type(labels) == list:
                labels = np.array(labels)
            labels = labels.astype(np.int32)

        self.data_count = len(datas)
        if labels is not None and len(labels.shape) == 1:
            self.class_num = labels.max() + 1
            label_matrix = np.zeros((self.data_count, self.class_num))
            label_matrix[list(range(self.data_count)), labels] = 1
            labels = label_matrix

        self.batch_size = batch_size
        self.datas = datas
        self.batch_count = math.ceil((self.data_count - 1) / batch_size)
        self.labels = labels
        self.reset()

    def reset(self):
        self.next_batch_index = 0

    def has_next(self):
        return self.next_batch_index < self.batch_count

    def __next__(self):
        if not self.has_next():
            return None
        next_data_batch = self.datas[self.next_batch_index::self.batch_count]
        if self.labels is None:
            result = [next_data_batch, None]
        else:
            next_label_batch = self.labels[self.next_batch_index::self.batch_count]
            result = [next_data_batch, next_label_batch]
        self.next_batch_index += 1
        return result


def default_cost_listener(epoch, batch_index, cost_value):
    if epoch % 20 == 0 and batch_index % 10 == 0:
        print("epoch: {} batch: {} cost: {}".format(epoch, batch_index, cost_value))


class AutoEncoder(object):
    def __init__(self, cost_listener=default_cost_listener):
        self.cost_listener = cost_listener

    def fine_tune(self,
                  ite,
                  learning_rate=0.01,
                  max_epoch=1000,
                  supervised=False,
                  corrupt=0,
                  tied=False
                  ):
        if supervised:
            self.supervised_fine_tune(ite, learning_rate, max_epoch, corrupt, tied)
        else:
            self.unsupervised_fine_tune(ite, learning_rate, max_epoch, corrupt, tied)

    def unsupervised_fine_tune(self, ite, learning_rate, max_epoch, corrupt, tied):
        print("\nstart unsupervised fine tuning ============\n")
        self.sess.close()
        self.init_tf_vars(reuse=True, tied=tied)
        self.build_base_structure()
        self.optimize_cost(ite, learning_rate, max_epoch, corrupt)
        self.save_unstacked_params()

    def supervised_fine_tune(self, ite, learning_rate, max_epoch, corrupt, tied):
        print("\nstart supervised fine tuning ============\n")
        self.sess.close()
        self.init_tf_vars(reuse=True, tied=tied)
        self.build_base_structure()

        ite.reset()
        peek_batch = next(ite)
        class_num = peek_batch[1].shape[1]
        ite.reset()

        self.fine_tune_outputs = self.build_fine_tuning(class_num)
        fine_tune_outputs_holder = tf.placeholder(tf.float32, [None, class_num])
        fine_tune_cost = tf.reduce_mean(-tf.reduce_sum(fine_tune_outputs_holder * tf.log(tf.clip_by_value(self.fine_tune_outputs, 1e-10, 1.0)), reduction_indices=1))
        fine_tune_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(fine_tune_cost)

        self.init_session()

        for epoch in range(max_epoch):
            ite.reset()
            while ite.has_next():
                batch_index = ite.next_batch_index
                batch_inputs, batch_labels = next(ite)
                batch_outputs = batch_inputs
                if corrupt > 0:
                    batch_inputs = self.corrupt_inputs(batch_inputs, corrupt)
                feed_dict = {
                    self.inputs_holder: batch_inputs,
                    fine_tune_outputs_holder: batch_labels,
                    self.outputs_holder: batch_outputs
                }
                _, c = self.sess.run([fine_tune_optimizer, fine_tune_cost], feed_dict=feed_dict)
                self.cost_listener(epoch, batch_index, c)

        self.save_unstacked_params()

    # neuron_nums is the number of neuron for each hidden layer
    # the number of neuron of input layer and decoder is excluded
    def fit(self,
            neuron_nums,
            ite,
            learning_rate=0.01,
            max_epoch=1000,
            stacked=False,
            activation="tanh",
            corrupt=0,
            tied=False
            ):
        # self.stacked = stacked
        self.activation = activation
        if stacked:
            self.stacked_fit(neuron_nums, ite, learning_rate, max_epoch, corrupt, tied)
        else:
            self.unstacked_fit(neuron_nums, ite, learning_rate, max_epoch, corrupt, tied)

    def stacked_fit(self, neuron_nums, ite, learning_rate, max_epoch, corrupt, tied):
        # self.activation = "sigmoid"
        self.ws = neuron_nums

        ite.reset()
        peek_batch = next(ite)
        self.input_dim = peek_batch[0].shape[1]
        ite.reset()

        self.ws = [self.input_dim] + self.ws

        autoencoders = []

        current_ite = ite
        # train each layer greedily
        for i in range(1, len(self.ws)):
            print("\ntraining layer {}\n".format(i))
            current_autoencoder = AutoEncoder(cost_listener=self.cost_listener)
            if type(max_epoch) == list:
                current_max_epoch = max_epoch[i - 1]
            else:
                current_max_epoch = max_epoch
            current_autoencoder.fit([self.ws[i]], current_ite, activation=self.activation,
                                    learning_rate=learning_rate, max_epoch=current_max_epoch, corrupt=corrupt, tied=tied)
            current_outputs = None

            current_ite.reset()
            while current_ite.has_next():
                # batch_index = ite.next_batch_index
                batch_inputs, _ = next(current_ite)
                batch_outputs = current_autoencoder.encode(batch_inputs)
                if current_outputs is None:
                    current_outputs = batch_outputs
                else:
                    current_outputs = np.concatenate((current_outputs, batch_outputs), 0)
            print("************")
            print(current_outputs)
            current_ite = DataIterator(current_outputs)
            current_autoencoder.close()
            autoencoders.append(current_autoencoder)
        self.save_stacked_params(autoencoders)

        self.init_tf_vars(reuse=True, tied=tied)
        self.build_base_structure()
        self.init_session()

    def init_session(self):
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def build_base_structure(self):
        self.inputs_holder = tf.placeholder(tf.float32, [None, self.ws[0]])
        # self.keep_prob = tf.placeholder(tf.float32)
        self.outputs_holder = tf.placeholder(tf.float32, [None, self.ws[0]])
        self.encoder = self.build_encoder(self.inputs_holder)
        self.generator_inputs_holder = tf.placeholder(tf.float32, [None, self.ws[-1]])
        self.generator = self.build_decoder(self.generator_inputs_holder)
        self.decoder = self.build_decoder(self.encoder)

    def corrupt_inputs(self, inputs, corrupt):
        # noise_mask = np.random.rand(inputs.shape[0], inputs.shape[1])
        # noise_mask[noise_mask > corrupt] = 0
        corrupt_num = int(inputs.shape[0] * inputs.shape[1] * corrupt)
        corrupt_rows = np.random.randint(0, inputs.shape[0], corrupt_num)
        corrupt_cols = np.random.randint(0, inputs.shape[1], corrupt_num)
        corrupted_inputs = inputs.copy()
        corrupted_inputs[corrupt_rows, corrupt_cols] = 0
        return corrupted_inputs
        # return inputs + noise_mask - corrupt/2

    def optimize_cost(self, ite, learning_rate, max_epoch, corrupt):
        cost = tf.reduce_mean(tf.pow(self.decoder - self.outputs_holder, 2))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        self.init_session()
        for epoch in range(max_epoch):
            ite.reset()
            while ite.has_next():
                batch_index = ite.next_batch_index
                batch_inputs, _ = next(ite)
                batch_outputs = batch_inputs
                if corrupt > 0:
                    batch_inputs = self.corrupt_inputs(batch_inputs, corrupt)
                feed_dict = {
                    self.inputs_holder: batch_inputs,
                    self.outputs_holder: batch_outputs
                }
                _, c = self.sess.run([optimizer, cost], feed_dict=feed_dict)
                self.cost_listener(epoch, batch_index, c)

    def unstacked_fit(self, neuron_nums, ite, learning_rate, max_epoch, corrupt, tied):
        self.ws = neuron_nums
        ite.reset()
        peek_batch = next(ite)
        self.input_dim = peek_batch[0].shape[1]
        ite.reset()
        self.ws = [self.input_dim] + self.ws
        self.init_tf_vars(reuse=False, tied=tied)

        self.build_base_structure()
        print("\nstart training autoencoder ============\n")
        self.optimize_cost(ite, learning_rate, max_epoch, corrupt)
        self.save_unstacked_params()

    def save_stacked_params(self, autoencoders):
        self.params = {"encoder": [], "decoder": []}
        for autoencoder in autoencoders:
            self.params["encoder"].append(autoencoder.params["encoder"][0])
            self.params["decoder"].append(autoencoder.params["decoder"][-1])
        self.params["decoder"].reverse()

    def save_unstacked_params(self):
        self.params = {"encoder": [], "decoder": []}
        for encoder_var in self.tf_vars["encoder"]:
            W = self.sess.run(encoder_var["W"])
            b = self.sess.run(encoder_var["b"])
            self.params["encoder"].append({"W": W, "b": b})
        for decoder_var in self.tf_vars["decoder"]:
            W = self.sess.run(decoder_var["W"])
            b = self.sess.run(decoder_var["b"])
            self.params["decoder"].append({"W": W, "b": b})

    def init_tf_vars(self, reuse, tied):
        self.tf_vars = {}
        self.tf_vars["encoder"] = []
        self.tf_vars["decoder"] = []

        for i in range(len(self.ws) - 1):
            if not reuse:
                W = tf.Variable(tf.truncated_normal([self.ws[i], self.ws[i + 1]]))
                b = tf.Variable(tf.truncated_normal([self.ws[i + 1]]))
            else:
                param = self.params["encoder"][i]
                W = tf.Variable(param["W"])
                b = tf.Variable(param["b"])
            self.tf_vars["encoder"].append({"W": W, "b": b})

        for i in range(1, len(self.ws)):
            if not reuse:
                if tied:
                    W = tf.transpose(self.tf_vars["encoder"][-i]["W"])
                else:
                    W = tf.Variable(tf.truncated_normal([self.ws[-i], self.ws[-(i + 1)]]))
                b = tf.Variable(tf.truncated_normal([self.ws[-(i + 1)]]))
            else:
                param = self.params["decoder"][i - 1]
                if tied:
                    W = tf.transpose(self.tf_vars["encoder"][-i]["W"])
                else:
                    W = tf.Variable(param["W"])
                b = tf.Variable(param["b"])
            self.tf_vars["decoder"].append({"W": W, "b": b})

    def build_encoder(self, inputs):
        encoder = inputs
        for i, param_var in enumerate(self.tf_vars["encoder"]):
            W = param_var["W"]
            b = param_var["b"]
            encoder = tf.matmul(encoder, W) + b
            if self.activation == "sigmoid":
                encoder = tf.nn.sigmoid(encoder)
            elif self.activation == "tanh":
                encoder = tf.nn.tanh(encoder)
            else:
                raise Exception('Invalid Activation Function "{}"'.format(self.activation))
        return encoder

    def build_decoder(self, inputs):
        decoder = inputs
        for i, param_var in enumerate(self.tf_vars["decoder"]):
            W = param_var["W"]
            b = param_var["b"]
            decoder = tf.matmul(decoder, W) + b
            if self.activation == "sigmoid":
                decoder = tf.nn.sigmoid(decoder)
            elif self.activation == "tanh":
                decoder = tf.nn.tanh(decoder)
            else:
                raise Exception('Invalid Activation Function "{}"'.format(self.activation))
        return decoder

    def encode(self, inputs):
        feed_dict = {self.inputs_holder: inputs}
        encoder_outputs = self.sess.run(self.encoder, feed_dict=feed_dict)
        return encoder_outputs

    def decode(self, inputs):
        feed_dict = {self.generator_inputs_holder: inputs}
        decoder_outputs = self.sess.run(self.generator, feed_dict=feed_dict)
        return decoder_outputs

    def predict(self, inputs):
        feed_dict = {self.inputs_holder: inputs}
        predicted_outputs = self.sess.run(self.fine_tune_outputs, feed_dict=feed_dict)
        return predicted_outputs

    def reconstruct(self, inputs):
        feed_dict = {self.inputs_holder: inputs}
        reconstruct_outputs = self.sess.run(self.decoder, feed_dict=feed_dict)
        return reconstruct_outputs

    def build_fine_tuning(self, class_num):
        self.tf_vars["output"] = {}
        W = tf.Variable(tf.truncated_normal([self.ws[-1], class_num]), name="W")
        b = tf.Variable(tf.truncated_normal([class_num]))
        self.tf_vars["output"]["W"] = W
        self.tf_vars["output"]["b"] = b
        fine_tune_outputs = tf.matmul(self.encoder, W) + b
        fine_tune_outputs = tf.nn.softmax(fine_tune_outputs)
        return fine_tune_outputs

    def close(self):
        self.sess.close()


x, y = load_iris(True)
shuffle_index = np.arange(len(x))
np.random.shuffle(shuffle_index)
x = x[shuffle_index]
y = y[shuffle_index]

print(x.shape, y.shape)
datas = x
# normalize datas
datas = np.array(datas, dtype=np.float32)
datas = datas / datas.max(0)
datas = datas * 2 - 1

labels = y

# data wrapper
iterator = DataIterator(datas)
fine_tuning_iterator = DataIterator(datas, labels=labels)

# train autoencoder
# assume the input dimension is input_d
# the network is like input_d -> 4 -> 2 -> 4 -> input_d
autoencoder = AutoEncoder()

# train autoencoder without fine-tuning
print("\ntrain autoencoder without fine-tuning ==========\n")
autoencoder.fit([4, 2], iterator, stacked=True, learning_rate=0.02, max_epoch=5000, tied=True, activation="tanh")

# encode data (without fine-tuning)
encoded_datas = autoencoder.encode(datas)
print("encoder (without fine-tuning) ================")
print(encoded_datas)

# train autoencoder with fine-tuning
print("\ntrain autoencoder with fine-tuning ==========\n")
autoencoder.fine_tune(fine_tuning_iterator, supervised=True, learning_rate=0.02, max_epoch=10000, tied=True)
# autoencoder.fine_tune(fine_tuning_iterator, supervised = False, learning_rate = 0.02, max_epoch = 6000)

# encode data (with fine-tuning)
tuned_encoded_datas = autoencoder.encode(datas)
print("encoder (with fine-tuning)================")
print(tuned_encoded_datas)

# predict data( based on fine tuning )
predicted_datas = autoencoder.predict(datas)
print("predicted ================")
print(predicted_datas)
predicted_labels = predicted_datas.argmax(1)
eval_array = (predicted_labels == labels)
correct_count = len(np.where(eval_array == True)[0])
error_count = len(np.where(eval_array == False)[0])
correct_rate = float(correct_count) / (correct_count + error_count)
error_rate = float(error_count) / (correct_count + error_count)
print("correct: {}({})\terror: {}({})".format(correct_count, "%.2f" % correct_rate, error_count, "%.2f" % error_rate))
autoencoder.close()

# visualize encoded datas
colors = ["red", "green", "blue"]
label_colors = [colors[label_id] for label_id in labels]

fig_3d = plt.figure("origin iris data")
plot_3d = fig_3d.add_subplot(111, projection='3d')
plot_3d.scatter(datas[:, 0], datas[:, 1], datas[:, 2], color=label_colors)

fig_2d = plt.figure("encoded iris data (without fine-tuning)")
plot_2d = fig_2d.add_subplot(111)
plot_2d.scatter(encoded_datas[:, 0], encoded_datas[:, 1], color=label_colors)

fig_tuned_2d = plt.figure("encoded iris data (with fine-tuning)")
plot_tuned_2d = fig_tuned_2d.add_subplot(111)
plot_tuned_2d.scatter(tuned_encoded_datas[:, 0], tuned_encoded_datas[:, 1], color=label_colors)

plt.show()
