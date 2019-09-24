# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import six
import tensorflow as tf
import numpy as np


class BertConfig(object):
    """Configuration for `BertModel`."""

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,

                 # dcnn variable
                 repeat_block=False,
                 num_cnn_layers=9,
                 filter_width=3
                 ):
        """Constructs BertConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler.
            hidden_dropout_prob: The dropout probability for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The stdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

        # dcnn config
        self.repeat_block = repeat_block
        self.num_cnn_layers = num_cnn_layers
        self.filter_width = filter_width

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertModel(object):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Example usage:

    ```python
    # Already been converted into WordPiece token ids
    input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
    input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
    token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config, is_training=True,
        input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

    label_embeddings = tf.get_variable(...)
    pooled_output = model.get_pooled_output()
    logits = tf.matmul(pooled_output, label_embeddings)
    ...
    ```
    """

    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=True,
                 scope=None):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
            is_training: bool. rue for training model, false for eval model. Controls
                whether dropout will be applied.
            input_ids: int32 Tensor of shape [batch_size, seq_length].
            input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
            token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
            use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
                embeddings or tf.embedding_lookup() for the word embeddings. On the TPU,
                it is must faster if this is True, on the CPU or GPU, it is faster if
                this is False.
            scope: (optional) variable scope. Defaults to "bert".

        Raises:
            ValueError: The config is invalid or one of the input tensor shapes
                is invalid.
        """
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        with tf.variable_scope(scope, default_name="bert"):
            with tf.variable_scope("embeddings"):
                # Perform embedding lookup on the word ids.
                (self.embedding_output, self.embedding_table) = embedding_lookup(
                        input_ids=input_ids,
                        vocab_size=config.vocab_size,
                        embedding_size=config.hidden_size,
                        initializer_range=config.initializer_range,
                        word_embedding_name="word_embeddings",
                        use_one_hot_embeddings=use_one_hot_embeddings)

                # Add positional embeddings and token type embeddings, then layer
                # normalize and perform dropout.
                self.embedding_output = embedding_postprocessor(
                        input_tensor=self.embedding_output,
                        use_token_type=True,
                        token_type_ids=token_type_ids,
                        token_type_vocab_size=config.type_vocab_size,
                        token_type_embedding_name="token_type_embeddings",
                        use_position_embeddings=True,
                        position_embedding_name="position_embeddings",
                        initializer_range=config.initializer_range,
                        max_position_embeddings=config.max_position_embeddings,
                        dropout_prob=config.hidden_dropout_prob)

            with tf.variable_scope("encoder"):
                # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
                # mask of shape [batch_size, seq_length, seq_length] which is used
                # for the attention scores.
                attention_mask = create_attention_mask_from_input_mask(input_ids, input_mask)

                # Run the stacked transformer.
                # `sequence_output` shape = [batch_size, seq_length, hidden_size].

                out = transformer_model(input_tensor=self.embedding_output,
                                        attention_mask=attention_mask,
                                        hidden_size=config.hidden_size,
                                        num_hidden_layers=config.num_hidden_layers,
                                        num_attention_heads=config.num_attention_heads,
                                        intermediate_size=config.intermediate_size,
                                        intermediate_act_fn=get_activation(config.hidden_act),
                                        hidden_dropout_prob=config.hidden_dropout_prob,
                                        attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                                        initializer_range=config.initializer_range,
                                        do_return_all_layers=True)

                self.all_encoder_layers = out[0]

            self.sequence_output = self.all_encoder_layers[-1]
            # The "pooler" converts the encoded sequence tensor of shape
            # [batch_size, seq_length, hidden_size] to a tensor of shape
            # [batch_size, hidden_size]. This is necessary for segment-level
            # (or segment-pair-level) classification tasks where we need a fixed
            # dimensional representation of the segment.
            with tf.variable_scope("pooler"):
                # We "pool" the model by simply taking the hidden state corresponding
                # to the first token. We assume that this has been pre-trained
                first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
                self.pooled_output = tf.layers.dense(
                        first_token_tensor,
                        config.hidden_size,
                        activation=tf.tanh,
                        kernel_initializer=create_initializer(config.initializer_range))

    def get_pooled_output(self):
        return self.pooled_output

    def get_sequence_output(self):
        """Gets final hidden layer of encoder.

        Returns:
            float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
            to the final hidden of the transformer encoder.
        """
        return self.sequence_output

    def get_all_encoder_layers(self):
        return self.all_encoder_layers

    def get_embedding_output(self):
        """Gets output of the embedding lookup (i.e., input to the transformer).

        Returns:
            float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
            to the output of the embedding layer, after summing the word
            embeddings with the positional embeddings and the token type embeddings,
            then performing layer normalization. This is the input to the transformer.
        """
        return self.embedding_output

    def get_embedding_table(self):
        return self.embedding_table


def gelu(input_tensor):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
        input_tensor: float Tensor to perform activation.

    Returns:
        `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

    Args:
        activation_string: String name of the activation function.

    Returns:
        A Python function corresponding to the activation function. If
        `activation_string` is None, empty, or "linear", this will return None.
        If `activation_string` is not a string, it will return `activation_string`.

    Raises:
        ValueError: The `activation_string` does not correspond to a known
            activation.
    """

    # We assume that anything that"s not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
        input_tensor: float Tensor.
        dropout_prob: Python float. The probability of dropping out a value (NOT of
            *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
        A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
            inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
    """Runs layer normalization followed by dropout."""
    output_tensor = layer_norm(input_tensor, name)
    output_tensor = dropout(output_tensor, dropout_prob)
    return output_tensor


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
    """Looks up words embeddings for id tensor.

    Args:
        input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
            ids.
        vocab_size: int. Size of the embedding vocabulary.
        embedding_size: int. Width of the word embeddings.
        initializer_range: float. Embedding initialization range.
        word_embedding_name: string. Name of the embedding table.
        use_one_hot_embeddings: bool. If True, use one-hot method for word
            embeddings. If False, use `tf.nn.embedding_lookup()`. One hot is better
            for TPUs.

    Returns:
        float Tensor of shape [batch_size, seq_length, embedding_size].
    """
    # This function assumes that the input is of shape [batch_size, seq_length,
    # num_inputs].
    #
    # If the input is a 2D tensor of shape [batch_size, seq_length], we
    # reshape to [batch_size, seq_length, 1].
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])

    embedding_table = tf.get_variable(name=word_embedding_name,
                                      shape=[vocab_size, embedding_size],
                                      initializer=create_initializer(initializer_range))

    if use_one_hot_embeddings:
        flat_input_ids = tf.reshape(input_ids, [-1])
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
        output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
        output = tf.nn.embedding_lookup(embedding_table, input_ids)

    input_shape = get_shape_list(input_ids)

    output = tf.reshape(output, input_shape[0:-1] + [input_shape[-1] * embedding_size])
    return (output, embedding_table)


def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
    """Performs various post-processing on a word embedding tensor.

    Args:
        input_tensor: float Tensor of shape [batch_size, seq_length,
            embedding_size].
        use_token_type: bool. Whether to add embeddings for `token_type_ids`.
        token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
            Must be specified if `use_token_type` is True.
        token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
        token_type_embedding_name: string. The name of the embedding table variable
            for token type ids.
        use_position_embeddings: bool. Whether to add position embeddings for the
            position of each token in the sequence.
        position_embedding_name: string. The name of the embedding table variable
            for positional embeddings.
        initializer_range: float. Range of the weight initialization.
        max_position_embeddings: int. Maximum sequence length that might ever be
            used with this model. This can be longer than the sequence length of
            input_tensor, but cannot be shorter.
        dropout_prob: float. Dropout probability applied to the final output tensor.

    Returns:
        float tensor with same shape as `input_tensor`.

    Raises:
        ValueError: One of the tensor shapes or input values is invalid.
    """
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = input_tensor

    if use_token_type:
        if token_type_ids is None:
            raise ValueError("`token_type_ids` must be specified if"
                                             "`use_token_type` is True.")
        token_type_table = tf.get_variable(
                name=token_type_embedding_name,
                shape=[token_type_vocab_size, width],
                initializer=create_initializer(initializer_range))
        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        flat_token_type_ids = tf.reshape(token_type_ids, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
        token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
        token_type_embeddings = tf.reshape(token_type_embeddings, [batch_size, seq_length, width])
        output += token_type_embeddings

    if use_position_embeddings:
        assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
        with tf.control_dependencies([assert_op]):
            full_position_embeddings = tf.get_variable(name=position_embedding_name,
                                                       shape=[max_position_embeddings, width],
                                                       initializer=create_initializer(initializer_range))
            # Since the position embedding table is a learned variable, we create it
            # using a (long) sequence length `max_position_embeddings`. The actual
            # sequence length might be shorter than this, for faster training of
            # tasks that do not have long sequences.
            #
            # So `full_position_embeddings` is effectively an embedding table
            # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
            # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
            # perform a slice.
            position_embeddings = tf.slice(full_position_embeddings, [0, 0], [seq_length, -1])
            num_dims = len(output.shape.as_list())

            # Only the last two dimensions are relevant (`seq_length` and `width`), so
            # we broadcast among the first dimensions, which is typically just
            # the batch size.
            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([seq_length, width])
            position_embeddings = tf.reshape(position_embeddings, position_broadcast_shape)
            output += position_embeddings

    output = layer_norm_and_dropout(output, dropout_prob)
    return output


def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.

    Args:
        from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
        to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
        float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask

    return mask


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
    """Performs multi-headed attention from `from_tensor` to `to_tensor`.

    This is an implementation of multi-headed attention based on "Attention
    is all you Need". If `from_tensor` and `to_tensor` are the same, then
    this is self-attention. Each timestep in `from_tensor` attends to the
    corresponding sequence in `to_tensor`, and returns a fixed-with vector.

    This function first projects `from_tensor` into a "query" tensor and
    `to_tensor` into "key" and "value" tensors. These are (effectively) a list
    of tensors of length `num_attention_heads`, where each tensor is of shape
    [batch_size, seq_length, size_per_head].

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor and returned.

    In practice, the multi-headed attention are done with transposes and
    reshapes rather than actual separate tensors.

    Args:
        from_tensor: float Tensor of shape [batch_size, from_seq_length,
            from_width].
        to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
        attention_mask: (optional) int32 Tensor of shape [batch_size,
            from_seq_length, to_seq_length]. The values should be 1 or 0. The
            attention scores will effectively be set to -infinity for any positions in
            the mask that are 0, and will be unchanged for positions that are 1.
        num_attention_heads: int. Number of attention heads.
        size_per_head: int. Size of each attention head.
        query_act: (optional) Activation function for the query transform.
        key_act: (optional) Activation function for the key transform.
        value_act: (optional) Activation function for the value transform.
        attention_probs_dropout_prob: (optional) float. Dropout probability of the
            attention probabilities.
        initializer_range: float. Range of the weight initializer.
        do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
            * from_seq_length, num_attention_heads * size_per_head]. If False, the
            output will be of shape [batch_size, from_seq_length, num_attention_heads
            * size_per_head].
        batch_size: (Optional) int. If the input is 2D, this might be the batch size
            of the 3D version of the `from_tensor` and `to_tensor`.
        from_seq_length: (Optional) If the input is 2D, this might be the seq length
            of the 3D version of the `from_tensor`.
        to_seq_length: (Optional) If the input is 2D, this might be the seq length
            of the 3D version of the `to_tensor`.

    Returns:
        float Tensor of shape [batch_size, from_seq_length,
            num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
            true, this will be of shape [batch_size * from_seq_length,
            num_attention_heads * size_per_head]).

    Raises:
        ValueError: Any of the arguments or tensor shapes are invalid.
    """

    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,  seq_length, width):
        output_tensor = tf.reshape(input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    if len(from_shape) != len(to_shape):
        raise ValueError("The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                    "When passing in rank 2 tensors to attention_layer, the values "
                    "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                    "must all be specified.")

    # Scalar dimensions referenced here:
    #     B = batch size (number of sequences)
    #     F = `from_tensor` sequence length
    #     T = `to_tensor` sequence length
    #     N = `num_attention_heads`
    #     H = `size_per_head`

    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # `query_layer` = [B*F, N*H]
    query_layer = tf.layers.dense(
            from_tensor_2d,
            num_attention_heads * size_per_head,
            activation=query_act,
            name="query",
            kernel_initializer=create_initializer(initializer_range))

    # `key_layer` = [B*T, N*H]
    key_layer = tf.layers.dense(
            to_tensor_2d,
            num_attention_heads * size_per_head,
            activation=key_act,
            name="key",
            kernel_initializer=create_initializer(initializer_range))

    # `value_layer` = [B*T, N*H]
    value_layer = tf.layers.dense(
            to_tensor_2d,
            num_attention_heads * size_per_head,
            activation=value_act,
            name="value",
            kernel_initializer=create_initializer(initializer_range))

    # `query_layer` = [B, N, F, H]
    query_layer = transpose_for_scores(query_layer, batch_size, num_attention_heads, from_seq_length, size_per_head)

    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads, to_seq_length, size_per_head)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape( value_layer, [batch_size, to_seq_length, num_attention_heads, size_per_head])

    # `value_layer` = [B, N, T, H]
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    if do_return_2d_tensor:
        # `context_layer` = [B*F, N*V]
        context_layer = tf.reshape(context_layer, [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        # `context_layer` = [B, F, N*V]
        context_layer = tf.reshape(context_layer, [batch_size, from_seq_length, num_attention_heads * size_per_head])

    return context_layer, attention_probs


def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False,
                      use_residual=True):
    """Multi-headed, multi-layer Transformer from "Attention is All You Need".

    This is almost an exact implementation of the original Transformer encoder.

    See the original paper:
    https://arxiv.org/abs/1706.03762

    Also see:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

    Args:
        input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
        attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
            seq_length], with 1 for positions that can be attended to and 0 in
            positions that should not be.
        hidden_size: int. Hidden size of the Transformer.
        num_hidden_layers: int. Number of layers (blocks) in the Transformer.
        num_attention_heads: int. Number of attention heads in the Transformer.
        intermediate_size: int. The size of the "intermediate" (a.k.a., feed
            forward) layer.
        intermediate_act_fn: function. The non-linear activation function to apply
            to the output of the intermediate/feed-forward layer.
        hidden_dropout_prob: float. Dropout probability for the hidden layers.
        attention_probs_dropout_prob: float. Dropout probability of the attention
            probabilities.
        initializer_range: float. Range of the initializer (stddev of truncated
            normal).
        do_return_all_layers: Whether to also return all layers or just the final
            layer.

    Returns:
        float Tensor of shape [batch_size, seq_length, hidden_size], the final
        hidden layer of the Transformer.

    Raises:
        ValueError: A Tensor shape or parameter is invalid.
    """
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))

    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    # The Transformer performs sum residuals on all layers so the input needs
    # to be the same as the hidden size.
    if input_width != hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" % (input_width, hidden_size))

    # We keep the representation as a 2D tensor to avoid re-shaping it back and
    # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
    # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
    # help the optimizer.
    prev_output = reshape_to_matrix(input_tensor)

    all_layer_outputs = []
    last_attention_probs = None
    for layer_idx in range(num_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_idx):
            layer_input = prev_output

            with tf.variable_scope("attention"):
                attention_heads = []
                with tf.variable_scope("self"):
                    attention_head, last_attention_probs = attention_layer(
                                                            from_tensor=layer_input,
                                                            to_tensor=layer_input,
                                                            attention_mask=attention_mask,
                                                            num_attention_heads=num_attention_heads,
                                                            size_per_head=attention_head_size,
                                                            attention_probs_dropout_prob=attention_probs_dropout_prob,
                                                            initializer_range=initializer_range,
                                                            do_return_2d_tensor=True,
                                                            batch_size=batch_size,
                                                            from_seq_length=seq_length,
                                                            to_seq_length=seq_length)
                    attention_heads.append(attention_head)

                attention_output = None
                if len(attention_heads) == 1:
                    attention_output = attention_heads[0]
                else:
                    # In the case where we have other sequences, we just concatenate
                    # them to the self-attention head before the projection.
                    attention_output = tf.concat(attention_heads, axis=-1)

                # Run a linear projection of `hidden_size` then add a residual
                # with `layer_input`.
                with tf.variable_scope("output"):
                    attention_output = tf.layers.dense(
                            attention_output,
                            hidden_size,
                            kernel_initializer=create_initializer(initializer_range))
                    attention_output = dropout(attention_output, hidden_dropout_prob)
                    if use_residual:
                        attention_output = layer_norm(attention_output + layer_input)

            # The activation is only applied to the "intermediate" hidden layer.
            with tf.variable_scope("intermediate"):
                intermediate_output = tf.layers.dense(
                        attention_output,
                        intermediate_size,
                        activation=intermediate_act_fn,
                        kernel_initializer=create_initializer(initializer_range))

            # Down-project back to `hidden_size` then add the residual.
            with tf.variable_scope("output"):
                layer_output = tf.layers.dense(
                        intermediate_output,
                        hidden_size,
                        kernel_initializer=create_initializer(initializer_range))
                layer_output = dropout(layer_output, hidden_dropout_prob)
                if use_residual:
                    layer_output = layer_norm(layer_output + attention_output)
                prev_output = layer_output
                all_layer_outputs.append(layer_output)

    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
        return final_outputs, last_attention_probs
    else:
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output, last_attention_probs


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
        tensor: A tf.Tensor object to find the shape of.
        expected_rank: (optional) int. The expected rank of `tensor`. If this is
            specified and the `tensor` has a different rank, and exception will be
            thrown.
        name: Optional name of the tensor for the error message.

    Returns:
        A list of dimensions of the shape of tensor. All static dimensions will
        be returned as python integers, and dynamic dimensions will be returned
        as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
        tensor: A tf.Tensor to check the rank of.
        expected_rank: Python integer or list of integers, expected rank.
        name: Optional name of the tensor for the error message.

    Raises:
        ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
                "For the tensor `%s` in scope `%s`, the actual rank "
                "`%d` (shape = %s) is not equal to the expected rank `%s`" %
                (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))



#########################
#### Added methods ######
#########################
class DCNNModel(object):
    """DCNN model ("Dilated CNN Document Encoder").

    Example usage:

    ```python
    # Already been converted into WordPiece token ids
    input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
    input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
    token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config, is_training=True,
        input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

    label_embeddings = tf.get_variable(...)
    pooled_output = model.get_pooled_output()
    logits = tf.matmul(pooled_output, label_embeddings)
    ...
    ```
    """

    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=False,
                 scope=None):
        """Constructor for DCNNModel.

        Args:
            config: `BertConfig` instance.
            is_training: bool. rue for training model, false for eval model. Controls
                whether dropout will be applied.
            input_ids: int32 Tensor of shape [batch_size, seq_length].
            input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
            use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
                embeddings or tf.embedding_lookup() for the word embeddings. On the TPU,
                it is must faster if this is True, on the CPU or GPU, it is faster if
                this is False.
            scope: (optional) variable scope. Defaults to "dcnn".

        Raises:
            ValueError: The config is invalid or one of the input tensor shapes
                is invalid.
        """
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0

        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        with tf.variable_scope(scope, default_name="bert"):
            with tf.variable_scope("embeddings"):
                # Perform embedding lookup on the word ids.
                (self.embedding_output, self.embedding_table) = \
                    embedding_lookup(input_ids=input_ids,
                                     vocab_size=config.vocab_size,
                                     embedding_size=config.hidden_size,
                                     initializer_range=config.initializer_range,
                                     word_embedding_name="word_embeddings",
                                     use_one_hot_embeddings=use_one_hot_embeddings)

                # Add positional embeddings and token type embeddings, then layer
                # normalize and perform dropout.
                self.embedding_output = embedding_postprocessor(
                    input_tensor=self.embedding_output,
                    use_token_type=True,
                    token_type_ids=token_type_ids,
                    token_type_vocab_size=config.type_vocab_size,
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=True,
                    position_embedding_name="position_embeddings",
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob)

            with tf.variable_scope("encoder"):
                self.all_encoder_layers = []
                if config.num_cnn_layers:
                    # Run the stacked transformer.
                    # `sequence_output` shape = [batch_size, seq_length, hidden_size].
                    with tf.variable_scope("dcnn"):
                        self.all_encoder_layers += dilated_cnn(input_tensor=self.embedding_output,
                                                               input_mask=input_mask,
                                                               filter_width=config.filter_width,
                                                               hidden_size=config.hidden_size,
                                                               num_hidden_layers=config.num_cnn_layers,
                                                               hidden_dropout_prob=config.hidden_dropout_prob,
                                                               initializer_range=config.initializer_range,
                                                               repeat_block=config.repeat_block,
                                                               do_return_all_layers=True)
                # if number of hidden layers is greater than zero than run bert
                if config.num_hidden_layers:
                    if config.num_cnn_layers:
                        layer_input = self.all_encoder_layers[-1]
                    else:
                        layer_input = self.embedding_output

                    attention_mask = create_attention_mask_from_input_mask(input_ids, input_mask)

                    out = transformer_model(input_tensor=layer_input,
                                            attention_mask=attention_mask,
                                            hidden_size=config.hidden_size,
                                            num_hidden_layers=config.num_hidden_layers,
                                            num_attention_heads=config.num_attention_heads,
                                            intermediate_size=config.intermediate_size,
                                            intermediate_act_fn=get_activation(config.hidden_act),
                                            hidden_dropout_prob=config.hidden_dropout_prob,
                                            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                                            initializer_range=config.initializer_range,
                                            do_return_all_layers=True)
                    self.all_encoder_layers += out[0]
                    self.attention_probs = out[1]
        self.sequence_output = self.all_encoder_layers[-1]

        # The "pooler" converts the encoded sequence tensor of shape
        # [batch_size, seq_length, hidden_size] to a tensor of shape
        # [batch_size, hidden_size]. This is necessary for segment-level
        # (or segment-pair-level) classification tasks where we need a fixed
        # dimensional representation of the segment.
        with tf.variable_scope("pooler"):
            # We "pool" the model by simply taking the hidden state corresponding
            # to the first token. We assume that this has been pre-trained
            first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
            self.pooled_output = tf.layers.dense(first_token_tensor,
                                                 config.hidden_size,
                                                 activation=tf.tanh,
                                                 kernel_initializer=create_initializer(config.initializer_range))


    def get_pooled_output(self):
        return self.pooled_output

    def get_attention_probs(self):
        return self.attention_probs

    def get_sequence_output(self):
        """Gets final hidden layer of encoder.

        Returns:
            float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
            to the final hidden of the transformer encoder.
        """
        return self.sequence_output

    def get_all_encoder_layers(self):
        return self.all_encoder_layers

    def get_embedding_output(self):
        """Gets output of the embedding lookup (i.e., input to the transformer).

        Returns:
            float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
            to the output of the embedding layer, after summing the word
            embeddings with the positional embeddings and the token type embeddings,
            then performing layer normalization. This is the input to the transformer.
        """
        return self.embedding_output

    def get_embedding_table(self):
        return self.embedding_table


def dilated_cnn(input_tensor,
                input_mask,
                num_hidden_layers=9,
                hidden_dropout_prob=0.1,
                hidden_size=256,
                filter_width=3,
                repeat_block=False,
                initializer_range=0.02,
                do_return_all_layers=False,
                use_residual=True):

    """Dilated CNN

       Args:
           input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
           input_mask: (optional) int32 Tensor of shape [batch_size, seq_length]
               with 1 for positions that can be attended to and 0 in
               positions that should not be.
           hidden_size: int. Hidden size of the layer.
           filter_width: int. Window size of the CNN
           num_hidden_layers: int. Number of stacked layers (blocks) of DCNN.
           hidden_dropout_prob: float. Dropout probability for the hidden layers.
           initializer_range: float. Range of the initializer (stddev of truncated
               normal).
           repeat_block:
           do_return_all_layers: Whether to also return all layers or just the final
               layer.

       Returns:
           float Tensor of shape [batch_size, seq_length, hidden_size], the final
           hidden layer of the Transformer.

       Raises:
           ValueError: A Tensor shape or parameter is invalid.
       """

    input_shape = get_shape_list(input_tensor, expected_rank=3)
    input_width = input_shape[2]

    # The Transformer performs sum residuals on all layers so the input needs
    # to be the same as the hidden size.
    if input_width != hidden_size and use_residual:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %  (input_width, hidden_size))

    prev_output = input_tensor
    all_layer_outputs = []
    mask = tf.to_float(tf.not_equal(input_mask, 0))

    # dilated cnn layers
    for layer_idx in range(num_hidden_layers):
        scope_idx = layer_idx if not repeat_block else layer_idx % 3
        reuse_block = repeat_block and layer_idx > 2
        with tf.variable_scope("layer_%d" % scope_idx, reuse=reuse_block):
            layer_input = prev_output
            with tf.variable_scope("conv"):
                rate = np.broadcast_to(2**layer_idx, (1,))
                # print(rate)
                layer_output = _cnn_layer(layer_input, mask, rate, hidden_size, filter_width, initializer_range)

                # dropout residual and batch norm
                layer_output = dropout(layer_output, hidden_dropout_prob)
                if use_residual and (
                      (repeat_block and layer_idx % 3 == 0) or (not repeat_block and layer_idx and layer_idx % 2 == 0)):
                    layer_output = layer_norm(layer_output + layer_input)

                # update variable
                prev_output = layer_output
                all_layer_outputs.append(layer_output)

    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
        return final_outputs
    else:
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output


def _cnn_layer(inputs, mask, rate, hidden_size, filter_width, initializer_range):
    with tf.variable_scope("convolution"):
        input_size = inputs.get_shape().as_list()[-1]
        shape = [filter_width, input_size, 2 * hidden_size]
        filter_v = tf.get_variable("filter", shape, initializer=create_initializer(initializer_range))
        bias_v = tf.get_variable("bias", [2 * hidden_size])
        output = tf.nn.convolution(inputs, filter_v, "SAME", dilation_rate=rate)
        output = tf.nn.bias_add(output, bias_v)
        gate, act = tf.split(output, 2, 2)
        output = tf.nn.sigmoid(gate) * act

        return output * mask[:, :, None]


##########################
##### Sequential Model ###
##########################
class SeqConfig(object):
    """Configuration for `BertModel`."""

    def __init__(self,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.5,
                 attention_probs_dropout_prob=0.5,
                 max_position_embeddings=640,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 repeat_block=False,
                 num_cnn_layers=9,
                 filter_width=3,
                 use_pretemb=True,
                 emb_vocab_size=125117,
                 freeze=False,
                 use_feat=True,
                 feat_vocab_size=7,
                 feat_emb_size=50,
                 use_elmo=True,
                 concat_elmo=False,
                 elmo_size=1024,
                 elmo_lnum=3,
                 use_edoc=True,
                 concat_edoc=True,
                 edoc_size=256,
                 edoc_lnum=2,
                 use_flair=True,
                 concat_flair=True,
                 flair_size=2048,
                 flair_lnum=2,
                 use_bert=True,
                 concat_bert=True,
                 bert_size=1024,
                 bert_lnum=4,
                 num_lstm_layers=0,
                 compress_input=None, #None | lstm | ffnn
                 use_residual=True,
                 use_char=True,
                 max_char_len=64,
                 vocab_char=88,
                 char_dim=25,
                 char_kernel_size=3,
                 char_filters=30,
                 char_pool_size=64
                 ):

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

        # dcnn config
        self.repeat_block = repeat_block
        self.num_cnn_layers = num_cnn_layers
        self.filter_width = filter_width

        # pretrained emb
        self.use_pretemb = use_pretemb
        self.emb_vocab_size = emb_vocab_size
        self.freeze = freeze

        # features
        self.use_feat = use_feat
        self.feat_vocab_size = feat_vocab_size
        self.feat_emb_size = feat_emb_size

        # encoders
        self.use_elmo = use_elmo
        self.use_elmo = concat_elmo
        self.elmo_size = elmo_size
        self.elmo_lnum = elmo_lnum

        self.use_edoc = use_edoc
        self.concat_edoc = concat_edoc
        self.edoc_size = edoc_size
        self.edoc_lnum = edoc_lnum

        self.use_flair = use_flair
        self.concat_flair = concat_flair
        self.flair_size = flair_size
        self.flair_lnum = flair_lnum

        self.use_bert = use_bert
        self.concat_bert = concat_bert
        self.bert_size = bert_size
        self.bert_lnum = bert_lnum

        # divers
        self.num_lstm_layers = num_lstm_layers
        self.compress_input = compress_input
        self.use_residual = use_residual

        # chars
        self.use_char = use_char,
        self.vocab_char = vocab_char,
        self.max_char_len = max_char_len,
        self.char_dim = char_dim,
        self.char_kernel_size = char_kernel_size,
        self.char_filters = char_filters,
        self.char_pool_size = char_pool_size

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class SeqModel(object):

    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask,
                 token_type_ids,
                 emb_matrix,
                 feat_ids,
                 elmo_emb,
                 edoc_emb,
                 use_token_type=False,
                 scope=None):
        """Constructor for DCNNModel.

        Args:
            config: `BertConfig` instance.
            is_training: bool. rue for training model, false for eval model. Controls
                whether dropout will be applied.
            input_ids: int32 Tensor of shape [batch_size, seq_length].
            input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
            use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
                embeddings or tf.embedding_lookup() for the word embeddings. On the TPU,
                it is must faster if this is True, on the CPU or GPU, it is faster if
                this is False.
            use_token_type: Whether to add segment ids
            scope: (optional) variable scope. Defaults to "bert".

        Raises:
            ValueError: The config is invalid or one of the input tensor shapes
                is invalid.
        """
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0

        input_shape = get_shape_list(input_mask, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        seq_len = tf.to_int32(tf.reduce_sum(input_mask, -1))
        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        with tf.variable_scope(scope, default_name="bert"):
            with tf.variable_scope("embeddings"):
                # aggregate various input features (like pretrained word emb, word shape, elmo and doclevel)
                self.embedding_output = aggregate_inputs(config=config,
                                                         seq_len=seq_len,
                                                         emb_ids=input_ids,
                                                         emb_matrix=emb_matrix,
                                                         feat_ids=feat_ids,
                                                         elmo_emb=elmo_emb,
                                                         edoc_emb=edoc_emb)

                # Add positional embeddings and token type embeddings, then layer
                # normalize and perform dropout.
                self.embedding_output = embedding_postprocessor(
                    input_tensor=self.embedding_output,
                    use_token_type=use_token_type,
                    token_type_ids=token_type_ids,
                    token_type_vocab_size=config.type_vocab_size,
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=config.num_hidden_layers,
                    position_embedding_name="position_embeddings",
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob)

            layer_input = self.embedding_output
            with tf.variable_scope("encoder"):
                self.all_encoder_layers = []

                if config.num_lstm_layers:
                    with tf.variable_scope("lstm"):
                        self.all_encoder_layers += lstm_encoder(input_tensor=layer_input,
                                                                seq_len=seq_len,
                                                                layer_num=config.num_lstm_layers,
                                                                hidden_size=config.hidden_size,
                                                                hidden_dropout_prob=config.hidden_dropout_prob,
                                                                initializer_range=config.initializer_range)

                if config.num_cnn_layers:
                    with tf.variable_scope("dcnn"):
                        self.all_encoder_layers += dilated_cnn(input_tensor=layer_input,
                                                               input_mask=input_mask,
                                                               filter_width=config.filter_width,
                                                               hidden_size=config.hidden_size,
                                                               num_hidden_layers=config.num_cnn_layers,
                                                               hidden_dropout_prob=config.hidden_dropout_prob,
                                                               initializer_range=config.initializer_range,
                                                               repeat_block=config.repeat_block,
                                                               do_return_all_layers=True,
                                                               use_residual=config.use_residual)

                    layer_input = self.all_encoder_layers[-1]

                # if number of hidden layers is greater than zero than run bert
                if config.num_hidden_layers:
                    attention_mask = create_attention_mask_from_input_mask(input_ids, input_mask)

                    out = transformer_model(input_tensor=layer_input,
                                            attention_mask=attention_mask,
                                            hidden_size=config.hidden_size,
                                            num_hidden_layers=config.num_hidden_layers,
                                            num_attention_heads=config.num_attention_heads,
                                            intermediate_size=config.intermediate_size,
                                            intermediate_act_fn=get_activation(config.hidden_act),
                                            hidden_dropout_prob=config.hidden_dropout_prob,
                                            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                                            initializer_range=config.initializer_range,
                                            do_return_all_layers=True,
                                            use_residual=config.use_residual)

                    self.all_encoder_layers += out[0]
                    self.attention_probs = out[1]
        self.sequence_output = self.all_encoder_layers[-1]

        # The "pooler" converts the encoded sequence tensor of shape
        # [batch_size, seq_length, hidden_size] to a tensor of shape
        # [batch_size, hidden_size]. This is necessary for segment-level
        # (or segment-pair-level) classification tasks where we need a fixed
        # dimensional representation of the segment.
        with tf.variable_scope("pooler"):
            # We "pool" the model by simply taking the hidden state corresponding
            # to the first token. We assume that this has been pre-trained
            first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
            self.pooled_output = tf.layers.dense(first_token_tensor,
                                                 config.hidden_size,
                                                 activation=tf.tanh,
                                                 kernel_initializer=create_initializer(config.initializer_range))

    def get_pooled_output(self):
        return self.pooled_output

    def get_sequence_output(self):
        """Gets final hidden layer of encoder.

        Returns:
            float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
            to the final hidden of the transformer encoder.
        """
        return self.sequence_output

    def get_all_encoder_layers(self):
        return self.all_encoder_layers

    def get_embedding_output(self):
        """Gets output of the embedding lookup (i.e., input to the transformer).

        Returns:
            float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
            to the output of the embedding layer, after summing the word
            embeddings with the positional embeddings and the token type embeddings,
            then performing layer normalization. This is the input to the transformer.
        """
        return self.embedding_output


def aggregate_inputs(config, seq_len, emb_ids, emb_matrix, feat_ids, elmo_emb, edoc_emb):
    context_emb_list = []

    # add word level features (word shape and/or pred position if srl)
    if config.use_pretemb:
        with tf.variable_scope("pretemb"):
            output_tensor, _ = embedding_pretrained(emb_ids, emb_matrix, freeze=config.freeze)
        context_emb_list.append(output_tensor)

    # add word level features (word shape and/or pred position if srl)
    if config.use_feat:
        with tf.variable_scope("feat"):
            output_tensor, _ = embedding_lookup(feat_ids, config.feat_vocab_size, config.feat_emb_size,
                                                use_one_hot_embeddings=True)
        context_emb_list.append(output_tensor)

    # add elmo and edoc embeddings
    for enc in ["elmo", "edoc"]:
        if getattr(config, "use_%s"%enc):
            with tf.variable_scope(enc):
                context_emb_list.append(embedding_elmo(locals()["%s_emb"%enc], concat=getattr(config, "concat_%s"%enc)))

    context_emb = tf.concat(context_emb_list, 2)
    if config.compress_input == "ffnn":
        return tf.layers.dense(context_emb,
                                      config.hidden_size,
                                      activation=tf.tanh,
                                      kernel_initializer=create_initializer(config.initializer_range))
    elif config.compress_input == "lstm":
        return lstm_encoder(context_emb, seq_len, 1,
                            config.hidden_size, config.hidden_dropout_prob, config.initializer_range)

    return context_emb


def embedding_pretrained(input_ids,
                         emb_matrix,
                         word_embedding_name="pretrained_embeddings",
                         freeze=False):

    """Create an embeding table from pretrained word embeddings.

    Args:
        input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
            ids.
        vocab_size: int. Size of the embedding vocabulary.
        emb_matrix: numpy. pretrained word embedding matrix
        word_embedding_name: string. Name of the embedding table.
        use_one_hot_embeddings: bool. If True, use one-hot method for word
            embeddings. If False, use `tf.nn.embedding_lookup()`. One hot is better
            for TPUs.
        freeze: don't fine tune pretrained

    Returns:
        float Tensor of shape [batch_size, seq_length, embedding_size].
    """
    # This function assumes that the input is of shape [batch_size, seq_length,
    # num_inputs].
    #
    # If the input is a 2D tensor of shape [batch_size, seq_length], we
    # reshape to [batch_size, seq_length, 1].
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])

    embedding_table = tf.get_variable(name=word_embedding_name,
                                      shape=emb_matrix.shape,
                                      dtype=tf.float32,
                                      initializer=tf.constant_initializer(emb_matrix),
                                      trainable=not freeze)

    output = tf.nn.embedding_lookup(embedding_table, input_ids)
    input_shape = get_shape_list(input_ids)
    output = tf.reshape(output, input_shape[0:-1] + [input_shape[-1] * emb_matrix.shape[1]])

    return (output, embedding_table)


def embedding_elmo(lm_emb, concat=False):
    """Weight embeddings like elmo
           Args:
               lm_emb: float Tensor of shape [batch_size, max_seq_length, lm_num_layers, lm_emb_size].
           Returns:
               float Tensor of shape [batch_size, max_seq_length, lm_emb_size].
    """

    input_shape = get_shape_list(lm_emb)
    max_seq_length = input_shape[1]
    lm_emb_size = input_shape[2]
    lm_num_layers = input_shape[3]
    if concat:
        return tf.reshape(lm_emb, [-1, max_seq_length, lm_emb_size * lm_num_layers])

    with tf.variable_scope("lm_aggregation"):
        lm_weights = tf.nn.softmax(tf.get_variable("lm_scores", [lm_num_layers],
                                                   initializer=tf.constant_initializer(0.0)))
        lm_scaling = tf.get_variable("lm_scaling", [], initializer=tf.constant_initializer(1.0))

    flattened_lm_emb = tf.reshape(lm_emb, [-1, lm_num_layers])
    # [batch_size * max_seq_length * lm_emb_size, 1]
    flattened_aggregated_lm_emb = tf.matmul(flattened_lm_emb, tf.expand_dims(lm_weights, 1))
    aggregated_lm_emb = tf.reshape(flattened_aggregated_lm_emb, [-1, max_seq_length, lm_emb_size])
    aggregated_lm_emb *= lm_scaling

    return aggregated_lm_emb


def lstm_encoder(input_tensor, seq_len, layer_num, hidden_size,
                       hidden_dropout_prob, initializer_range):

    current_inputs = input_tensor # [batch_size, max_seq_length, emb]
    all_layer_outputs = []
    for layer in range(layer_num):
        with tf.variable_scope("layer_{}".format(layer)):
            with tf.variable_scope("fw_cell"):
                cell_fw = lstm_cell(hidden_size, hidden_dropout_prob)
            with tf.variable_scope("bw_cell"):
                cell_bw = lstm_cell(hidden_size, hidden_dropout_prob)

            (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                          cell_bw=cell_bw,
                                                                          inputs=current_inputs,
                                                                          sequence_length=seq_len,
                                                                          parallel_iterations=1024,
                                                                          swap_memory=False,
                                                                          dtype=tf.float32)

            text_outputs = tf.concat([fw_outputs, bw_outputs], 2) # [num_sentences, max_sentence_length, emb]

            if layer > 0:
                highway_gates = tf.layers.dense(text_outputs, 2 * hidden_size,
                                                kernel_initializer=create_initializer(initializer_range))
                highway_gates = tf.sigmoid(highway_gates)
                text_outputs = highway_gates * text_outputs + (1 - highway_gates) * current_inputs

            current_inputs = text_outputs
            all_layer_outputs.append(text_outputs)

    return all_layer_outputs


def lstm_cell(lstm_dim, hidden_dropout_prob):
    cell = tf.contrib.rnn.LSTMBlockCell(lstm_dim)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                         input_keep_prob=1-hidden_dropout_prob,
                                         output_keep_prob=1-hidden_dropout_prob)
    return cell


def char_cnn_model(config, char_ids):
    ntime_steps = tf.shape(char_ids)[1]

    char_input = tf.reshape(char_ids, [-1, config.max_char_len])
    char_embedding = tf.get_variable(name="char_embedding",
                                     dtype=tf.float32,
                                     initializer=tf.random_uniform([config.vocab_char,  config.char_dim], -0.5, 0.5),
                                     trainable=True)

    char_input = tf.nn.embedding_lookup(char_embedding, char_input)

    conv = tf.layers.conv1d(
        inputs=char_input,
        filters=config.char_filters,
        kernel_size=config.char_kernel_size,
        padding="same",
        activation=tf.nn.relu)

    pool = tf.layers.max_pooling1d(inputs=conv,
                                   pool_size=config.char_pool_size,
                                   strides=config.char_pool_size)

    output = tf.reshape(pool, shape=[-1, ntime_steps,
                                     config.char_filters * config.max_char_len//config.char_pool_size])

    return output
