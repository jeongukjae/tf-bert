import json

import tensorflow as tf


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
        **kwargs,  # unused
    ):
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.type_vocab_size = type_vocab_size
        self.vocab_size = vocab_size

    @staticmethod
    def from_json(path: str) -> "BertConfig":
        with open(path, "r") as f:
            file_content = json.load(f)

        return BertConfig(**file_content)


class Bert(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig):
        super().__init__()

        # embedding layer
        self.token_embeddings = tf.keras.layer.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = tf.keras.layer.Embedding(config.type_vocab_size, config.hidden_size)
        self.position_embeddings = tf.keras.layer.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embedding_layer_norm = tf.keras.layers.LayerNormalization()

        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, input_ids, position_ids, token_type_ids, attention_mask):
        words_embeddings = self.token_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + token_type_embeddings, position_embeddings
        embeddings = self.embedding_layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
