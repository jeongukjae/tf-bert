import json

import tensorflow as tf


def gelu(x):
    """Gaussian Error Linear Unit.

    Original paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * x * (1 + 0.044715 * x * x)))


def get_activation_function(hidden_act):
    if hidden_act == "linear":
        return None
    elif hidden_act == "relu":
        return tf.nn.relu
    elif hidden_act == "gelu":
        return gelu
    elif hidden_act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % hidden_act)


def get_initializer(x):
    return tf.keras.initializers.TruncatedNormal(stddev=x)


class BertConfig:
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 16,
        initializer_range: float = 0.0,
        layer_norm_eps: float = 1e-12,
        **kwargs,  # unused
    ):
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_act = get_activation_function(hidden_act)
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.type_vocab_size = type_vocab_size
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps

    @staticmethod
    def from_json(path: str) -> "BertConfig":
        with open(path, "r") as f:
            file_content = json.load(f)

        return BertConfig(**file_content)


class Bert(tf.keras.Model):
    def __init__(self, config: BertConfig):
        super().__init__()

        # embedding layer
        self.token_embeddings = tf.keras.layers.Embedding(
            config.vocab_size,
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
        )
        self.token_type_embeddings = tf.keras.layers.Embedding(
            config.type_vocab_size,
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
        )
        self.position_embeddings = tf.keras.layers.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
        )
        self.embedding_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, axis=-1
        )

        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

        # encoder
        self.encoders = [
            TransformerEncoder(config) for _ in range(config.num_hidden_layers)
        ]

        # pooler
        self.pooler_layer = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
        )

    def call(self, inputs):
        input_ids, token_type_ids, position_ids, attention_mask = inputs
        words_embeddings = self.token_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)
        attention_mask = (1.0 - attention_mask[:, tf.newaxis, :, tf.newaxis]) * -1e9

        embeddings = words_embeddings + token_type_embeddings + position_embeddings
        embeddings = self.embedding_layer_norm(embeddings)
        hidden_states = self.dropout(embeddings)

        for encoder in self.encoders:
            hidden_states = encoder(hidden_states, attention_mask)

        pooled_output = self.pooler_layer(hidden_states[:, 0, :])

        return hidden_states, pooled_output


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention_proj = tf.keras.layers.Dense(
            config.hidden_size * 3,
            kernel_initializer=get_initializer(config.initializer_range),
        )
        self.attention_dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
        )
        self.attention_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, axis=-1
        )

        self.intermediate_dense = tf.keras.layers.Dense(
            config.intermediate_size,
            kernel_initializer=get_initializer(config.initializer_range),
        )
        self.intermediate_act = config.hidden_act
        self.intermediate_dense2 = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
        )
        self.intermediate_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, axis=-1
        )

        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

        self.scaling_factor = float(config.num_attention_heads) ** -0.5
        self.num_head = config.num_attention_heads
        self.attention_depth = int(config.hidden_size / self.num_head)
        self.hidden_size = config.hidden_size

    def call(self, sequence, attention_mask):
        # multihead attention
        attention, _ = self._multihead_attention(sequence, attention_mask)
        # add and norm
        attention = self.attention_layer_norm(attention + sequence)
        # fc
        intermediate = self.intermediate_dense(attention)
        if self.intermediate_act is not None:
            intermediate = self.intermediate_act(intermediate)
        intermediate = self.dropout(self.intermediate_dense2(intermediate))
        # add and norm
        intermediate = self.intermediate_layer_norm(intermediate + attention)
        return intermediate

    def _multihead_attention(self, sequence, attention_mask):
        q, k, v = tf.split(self.attention_proj(sequence), num_or_size_splits=3, axis=-1)

        q = self._reshape_qkv(q)
        k = self._reshape_qkv(k)
        v = self._reshape_qkv(v)

        # calculate attention
        q *= self.scaling_factor
        attention = tf.matmul(q, k, transpose_b=True)
        attention_weight = tf.nn.softmax(attention, axis=-1)
        attention = tf.matmul(attention_weight, v)
        attention += attention_mask

        # concat
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        new_shape = [-1, tf.shape(attention)[1], self.hidden_size]
        attention = tf.reshape(attention, new_shape)

        # last dense net
        attention = self.attention_dense(attention)
        attention = self.dropout(attention)
        return attention, attention_weight

    def _reshape_qkv(self, val):
        new_shape = [-1, tf.shape(val)[1], self.num_head, self.attention_depth]
        return tf.transpose(tf.reshape(val, new_shape), perm=[0, 2, 1, 3])
