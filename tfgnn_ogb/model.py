from typing import Any, Dict, Tuple

import tensorflow_gnn as tfgnn
import tensorflow as tf


class NodeClassificationModel(tf.keras.Model):
    """
    Extends a Keras model to read the labels of the node to classify at
    training and evaluation time.
    """

    def __init__(self, *args: Tuple[Any], **kwargs: Dict[str, Any]) -> None:
        self.target_node = kwargs.pop("target_node", "paper")
        self.label_name = kwargs.pop("label_name", "label")
        super().__init__(*args, **kwargs)
        self.readout_labels = tfgnn.keras.layers.ReadoutFirstNode(
            node_set_name=self.target_node, feature_name=self.label_name
        )

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, data: tf.data.Dataset) -> Dict[str, float]:
        y = self.readout_labels(data)

        with tf.GradientTape() as tape:
            y_pred = self(data, training=True)
            loss = self.compiled_loss(
                y,
                y_pred,
                regularization_losses=self.losses,
            )

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, data: tf.data.Dataset) -> Dict[str, float]:
        y = self.readout_labels(data)
        y_pred = self(data, training=False)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


def build_model(num_classes: int, type_spec: tfgnn.GraphTensorSpec) -> tf.keras.Model:
    input = tf.keras.layers.Input(type_spec=type_spec)
    conv_1 = tfgnn.keras.ConvGNNBuilder(
        lambda edge: tfgnn.keras.layers.SimpleConvolution(tf.keras.layers.Dense(16)),
        lambda node: tfgnn.keras.layers.NextStateFromConcat(tf.keras.layers.Dense(16)),
    )
    conv_2 = tfgnn.keras.ConvGNNBuilder(
        lambda edge: tfgnn.keras.layers.SimpleConvolution(tf.keras.layers.Dense(32)),
        lambda node: tfgnn.keras.layers.NextStateFromConcat(tf.keras.layers.Dense(32)),
    )

    hidden = conv_1.Convolve()(input)
    hidden = conv_2.Convolve()(hidden)
    hidden = tfgnn.keras.layers.ReadoutFirstNode(node_set_name="paper")(hidden)
    output = tf.keras.layers.Dense(num_classes, activation="softmax")(hidden)
    return NodeClassificationModel(input, output)
