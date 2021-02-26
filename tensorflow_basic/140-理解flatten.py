"""
tensorflow.contrib.layers中提供了flatten函数
这个函数用于展平一个张量
(None,3,3,3)====> (None,27)
它保持第一维不变,其余维变化
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.layers import core as core_layers


def collect_named_outputs(collections, alias, outputs):
    """Add `Tensor` outputs tagged with alias to collections.

    It is useful to collect end-points or tags for summaries. Example of usage:

    logits = collect_named_outputs('end_points', 'inception_v3/logits', logits)
    assert 'inception_v3/logits' in logits.aliases

    Args:
      collections: A collection or list of collections. If None skip collection.
      alias: String to append to the list of aliases of outputs, for example,
             'inception_v3/conv1'.
      outputs: Tensor, an output tensor to collect

    Returns:
      The outputs Tensor to allow inline call.
    """
    if collections:
        append_tensor_alias(outputs, alias)
        ops.add_to_collections(collections, outputs)
    return outputs


def flatten(inputs, outputs_collections=None, scope=None):
    """Flattens the input while maintaining the batch_size.

      Assumes that the first dimension represents the batch.

    Args:
      inputs: A tensor of size [batch_size, ...].
      outputs_collections: Collection to add the outputs.
      scope: Optional scope for name_scope.

    Returns:
      A flattened tensor with shape [batch_size, k].
    Raises:
      ValueError: If inputs rank is unknown or less than 2.
    """
    with ops.name_scope(scope, 'Flatten', [inputs]) as sc:
        inputs = ops.convert_to_tensor(inputs)
        outputs = core_layers.flatten(inputs)
        return collect_named_outputs(outputs_collections, sc, outputs)


def flatten2(inputs):
    sz = np.prod(inputs.shape[1:])
    return tf.reshape(inputs, (-1, sz))


def flatten3(inputs):
    return tf.map_fn(lambda x: tf.reshape(x, (-1,)), inputs)


a = tf.placeholder(dtype=tf.int32, shape=(None, 3, 3))
print(a)
print(flatten(a).shape)
print(flatten2(a).shape)
print(flatten3(a).shape)
