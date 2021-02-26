import tensorflow as tf

"""
nce_loss真是万分复杂，如果彻底搞懂这个函数，会明白一系列函数
"""


def _compute_sampled_logits(weights,
                            biases,
                            labels,
                            inputs,
                            num_sampled,
                            num_classes,
                            num_true=1,
                            sampled_values=None,
                            subtract_log_q=True,
                            remove_accidental_hits=False,
                            partition_strategy="mod",
                            name=None,
                            seed=None):
    """Helper function for nce_loss and sampled_softmax_loss functions.

    Computes sampled output training logits and labels suitable for implementing
    e.g. noise-contrastive estimation (see nce_loss) or sampled softmax (see
    sampled_softmax_loss).

    Note: In the case where num_true > 1, we assign to each target class
    the target probability 1 / num_true so that the target probabilities
    sum to 1 per-example.

    Args:
      weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
          objects whose concatenation along dimension 0 has shape
          `[num_classes, dim]`.  The (possibly-partitioned) class embeddings.
      biases: A `Tensor` of shape `[num_classes]`.  The (possibly-partitioned)
          class biases.
      labels: A `Tensor` of type `int64` and shape `[batch_size,
          num_true]`. The target classes.  Note that this format differs from
          the `labels` argument of `nn.softmax_cross_entropy_with_logits_v2`.
      inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
          activations of the input network.
      num_sampled: An `int`.  The number of classes to randomly sample per batch.
      num_classes: An `int`. The number of possible classes.
      num_true: An `int`.  The number of target classes per training example.
      sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
          `sampled_expected_count`) returned by a `*_candidate_sampler` function.
          (if None, we default to `log_uniform_candidate_sampler`)
      subtract_log_q: A `bool`.  whether to subtract the log expected count of
          the labels in the sample to get the logits of the true labels.
          Default is True.  Turn off for Negative Sampling.
      remove_accidental_hits:  A `bool`.  whether to remove "accidental hits"
          where a sampled class equals one of the target classes.  Default is
          False.
      partition_strategy: A string specifying the partitioning strategy, relevant
          if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
          Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
      name: A name for the operation (optional).
      seed: random seed for candidate sampling. Default to None, which doesn't set
          the op-level random seed for candidate sampling.
    Returns:
      out_logits: `Tensor` object with shape
          `[batch_size, num_true + num_sampled]`, for passing to either
          `nn.sigmoid_cross_entropy_with_logits` (NCE) or
          `nn.softmax_cross_entropy_with_logits_v2` (sampled softmax).
      out_labels: A Tensor object with the same shape as `out_logits`.
    """

    if isinstance(weights, tf.Variable):
        weights = list(weights)
    if not isinstance(weights, list):
        weights = [weights]

    with tf.name_scope(name, "compute_sampled_logits",
                       weights + [biases, inputs, labels]):
        if labels.dtype != tf.int64:
            labels = tf.cast(labels, tf.int64)
        labels_flat = tf.reshape(labels, [-1])

        # Sample the negative labels.
        #   sampled shape: [num_sampled] tensor
        #   true_expected_count shape = [batch_size, 1] tensor
        #   sampled_expected_count shape = [num_sampled] tensor
        if sampled_values is None:
            sampled_values = tf.nn.log_uniform_candidate_sampler(
                true_classes=labels,
                num_true=num_true,
                num_sampled=num_sampled,
                unique=True,
                range_max=num_classes,
                seed=seed)
        # NOTE: pylint cannot tell that 'sampled_values' is a sequence
        # pylint: disable=unpacking-non-sequence
        sampled, true_expected_count, sampled_expected_count = (
            tf.stop_gradient(s) for s in sampled_values)
        # pylint: enable=unpacking-non-sequence
        sampled = tf.cast(sampled, tf.int64)

        # labels_flat is a [batch_size * num_true] tensor
        # sampled is a [num_sampled] int tensor
        all_ids = tf.concat([labels_flat, sampled], 0)

        # Retrieve the true weights and the logits of the sampled weights.

        # weights shape is [num_classes, dim]
        all_w = tf.nn.embedding_lookup(
            weights, all_ids, partition_strategy=partition_strategy)

        # true_w shape is [batch_size * num_true, dim]
        true_w = tf.slice(all_w, [0, 0],
                          tf.stack(
                              [tf.shape(labels_flat)[0], -1]))

        sampled_w = tf.slice(
            all_w, tf.stack([tf.shape(labels_flat)[0], 0]), [-1, -1])
        # inputs has shape [batch_size, dim]
        # sampled_w has shape [num_sampled, dim]
        # Apply X*W', which yields [batch_size, num_sampled]
        sampled_logits = tf.matmul(inputs, sampled_w, transpose_b=True)

        # Retrieve the true and sampled biases, compute the true logits, and
        # add the biases to the true and sampled logits.
        all_b = tf.nn.embedding_lookup(
            biases, all_ids, partition_strategy=partition_strategy)
        # true_b is a [batch_size * num_true] tensor
        # sampled_b is a [num_sampled] float tensor
        true_b = tf.slice(all_b, [0], tf.shape(labels_flat))
        sampled_b = tf.slice(all_b, tf.shape(labels_flat), [-1])

        # inputs shape is [batch_size, dim]
        # true_w shape is [batch_size * num_true, dim]
        # row_wise_dots is [batch_size, num_true, dim]
        dim = tf.shape(true_w)[1:2]
        new_true_w_shape = tf.concat([[-1, num_true], dim], 0)
        row_wise_dots = tf.multiply(
            tf.expand_dims(inputs, 1),
            tf.reshape(true_w, new_true_w_shape))
        # We want the row-wise dot plus biases which yields a
        # [batch_size, num_true] tensor of true_logits.
        dots_as_matrix = tf.reshape(row_wise_dots,
                                    tf.concat([[-1], dim], 0))
        true_logits = tf.reshape(_sum_rows(dots_as_matrix), [-1, num_true])
        true_b = tf.reshape(true_b, [-1, num_true])
        true_logits += true_b
        sampled_logits += sampled_b

        if remove_accidental_hits:
            acc_hits = tf.nn.compute_accidental_hits(
                labels, sampled, num_true=num_true)
            acc_indices, acc_ids, acc_weights = acc_hits

            # This is how SparseToDense expects the indices.
            acc_indices_2d = tf.reshape(acc_indices, [-1, 1])
            acc_ids_2d_int32 = tf.reshape(
                tf.cast(acc_ids, tf.int32), [-1, 1])
            sparse_indices = tf.concat([acc_indices_2d, acc_ids_2d_int32], 1,
                                       "sparse_indices")
            # Create sampled_logits_shape = [batch_size, num_sampled]
            sampled_logits_shape = tf.concat(
                [tf.shape(labels)[:1],
                 tf.expand_dims(num_sampled, 0)], 0)
            if sampled_logits.dtype != acc_weights.dtype:
                acc_weights = tf.cast(acc_weights, sampled_logits.dtype)
            sampled_logits += tf.sparse_to_dense(
                sparse_indices,
                sampled_logits_shape,
                acc_weights,
                default_value=0.0,
                validate_indices=False)

        if subtract_log_q:
            # Subtract log of Q(l), prior probability that l appears in sampled.
            true_logits -= tf.log(true_expected_count)
            sampled_logits -= tf.log(sampled_expected_count)

        # Construct output logits and labels. The true labels/logits start at col 0.
        out_logits = tf.concat([true_logits, sampled_logits], 1)

        # true_logits is a float tensor, ones_like(true_logits) is a float
        # tensor of ones. We then divide by num_true to ensure the per-example
        # labels sum to 1.0, i.e. form a proper probability distribution.
        out_labels = tf.concat([
            tf.ones_like(true_logits) / num_true,
            tf.zeros_like(sampled_logits)
        ], 1)

        return out_logits, out_labels


def _sum_rows(x):
    """Returns a vector summing up each row of the matrix x."""
    # _sum_rows(x) is equivalent to math_ops.reduce_sum(x, 1) when x is
    # a matrix.  The gradient of _sum_rows(x) is more efficient than
    # reduce_sum(x, 1)'s gradient in today's implementation. Therefore,
    # we use _sum_rows(x) in the nce_loss() computation since the loss
    # is mostly used for training.
    cols = tf.shape(x)[1]
    ones_shape = tf.stack([cols, 1])
    ones = tf.ones(ones_shape, x.dtype)
    return tf.reshape(tf.matmul(x, ones), [-1])


def nce_loss(weights,
             biases,
             labels,
             inputs,
             num_sampled,
             num_classes,
             num_true=1,
             sampled_values=None,
             remove_accidental_hits=False,
             partition_strategy="mod",
             name="nce_loss"):
    """Computes and returns the noise-contrastive estimation training loss.

    See [Noise-contrastive estimation: A new estimation principle for
    unnormalized statistical
    models](http://www.jmlr.org/proceedings/papers/v9/gutmann10a/gutmann10a.pdf).
    Also see our [Candidate Sampling Algorithms
    Reference](https://www.tensorflow.org/extras/candidate_sampling.pdf)

    A common use case is to use this method for training, and calculate the full
    sigmoid loss for evaluation or inference. In this case, you must set
    `partition_strategy="div"` for the two losses to be consistent, as in the
    following example:

    ```python
    if mode == "train":
      loss = tf.nn.nce_loss(
          weights=weights,
          biases=biases,
          labels=labels,
          inputs=inputs,
          ...,
          partition_strategy="div")
    elif mode == "eval":
      logits = tf.matmul(inputs, tf.transpose(weights))
      logits = tf.nn.bias_add(logits, biases)
      labels_one_hot = tf.one_hot(labels, n_classes)
      loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels_one_hot,
          logits=logits)
      loss = tf.reduce_sum(loss, axis=1)
    ```

    Note: By default this uses a log-uniform (Zipfian) distribution for sampling,
    so your labels must be sorted in order of decreasing frequency to achieve
    good results.  For more details, see
    `tf.nn.log_uniform_candidate_sampler`.

    Note: In the case where `num_true` > 1, we assign to each target class
    the target probability 1 / `num_true` so that the target probabilities
    sum to 1 per-example.

    Note: It would be useful to allow a variable number of target classes per
    example.  We hope to provide this functionality in a future release.
    For now, if you have a variable number of target classes, you can pad them
    out to a constant number by either repeating them or by padding
    with an otherwise unused class.

    Args:
      weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
          objects whose concatenation along dimension 0 has shape
          [num_classes, dim].  The (possibly-partitioned) class embeddings.
      biases: A `Tensor` of shape `[num_classes]`.  The class biases.
      labels: A `Tensor` of type `int64` and shape `[batch_size,
          num_true]`. The target classes.
      inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
          activations of the input network.
      num_sampled: An `int`.  The number of negative classes to randomly sample
          per batch. This single sample of negative classes is evaluated for each
          element in the batch.
      num_classes: An `int`. The number of possible classes.
      num_true: An `int`.  The number of target classes per training example.
      sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
          `sampled_expected_count`) returned by a `*_candidate_sampler` function.
          (if None, we default to `log_uniform_candidate_sampler`)
      remove_accidental_hits:  A `bool`.  Whether to remove "accidental hits"
          where a sampled class equals one of the target classes.  If set to
          `True`, this is a "Sampled Logistic" loss instead of NCE, and we are
          learning to generate log-odds instead of log probabilities.  See
          our [Candidate Sampling Algorithms Reference]
          (https://www.tensorflow.org/extras/candidate_sampling.pdf).
          Default is False.
      partition_strategy: A string specifying the partitioning strategy, relevant
          if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
          Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
      name: A name for the operation (optional).

    Returns:
      A `batch_size` 1-D tensor of per-example NCE losses.
    """
    logits, labels = _compute_sampled_logits(
        weights=weights,
        biases=biases,
        labels=labels,
        inputs=inputs,
        num_sampled=num_sampled,
        num_classes=num_classes,
        num_true=num_true,
        sampled_values=sampled_values,
        subtract_log_q=True,
        remove_accidental_hits=remove_accidental_hits,
        partition_strategy=partition_strategy,
        name=name)
    sampled_losses = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits, name="sampled_losses")
    # sampled_losses is batch_size x {true_loss, sampled_losses...}
    # We sum out true and sampled losses.
    return _sum_rows(sampled_losses)
