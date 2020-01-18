import pytest
import tensorflow as tf

from tf_bert.modeling import Bert, BertConfig


@pytest.fixture
def config():
    return BertConfig(300, num_hidden_layers=3)


@pytest.mark.parametrize("batch_size", [pytest.param(1), pytest.param(4)])
def test_bert_with_random_input(config: BertConfig, batch_size):
    model = Bert(config)

    input_ids = tf.random.uniform(
        (batch_size, config.max_position_embeddings),
        minval=0,
        maxval=300,
        dtype=tf.int32,
    )
    token_type_ids = tf.random.uniform(
        (batch_size, config.max_position_embeddings), minval=0, maxval=2, dtype=tf.int32
    )
    position_ids = tf.tile(
        tf.expand_dims(tf.range(0, config.max_position_embeddings, 1), 0),
        [batch_size, 1],
    )
    attention_mask = tf.random.uniform(
        (batch_size, config.max_position_embeddings), minval=0, maxval=2, dtype=tf.int32
    )
    attention_mask = tf.cast(attention_mask, dtype=tf.float32)

    encoder_ouputs, pooled_output = model(
        input_ids, token_type_ids, position_ids, attention_mask
    )

    assert encoder_ouputs.shape == (
        batch_size,
        config.max_position_embeddings,
        config.hidden_size,
    )
    assert pooled_output.shape == (
        batch_size,
        config.hidden_size,
    )
