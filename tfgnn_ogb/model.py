from typing import Any, Dict

import tensorflow_gnn as tfgnn
import tensorflow as tf


class NodeClassificationModel(tf.keras.Model):
    """
    Extends a Keras model to read the labels of the node to classify at
    training and evaluation time.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.target_node = kwargs.pop("target_node", "paper")
        self.label_name = kwargs.pop("label_name", "label")
        super().__init__(*args, **kwargs)

    def readout_labels(self, graph: tfgnn.GraphTensor) -> tf.Tensor:
        return tfgnn.gather_first_node(
            graph, node_set_name=self.target_node, feature_name=self.label_name
        )

    @tf.function
    def train_step(self, data: tf.data.Dataset) -> Dict[str, float]:
        y = self.readout_labels(data)

        with tf.GradientTape() as tape:
            y_pred = self(data, training=True)
            loss = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses,
            )

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data: tf.data.Dataset) -> Dict[str, float]:
        y = self.readout_labels(data)
        y_pred = self(data, training=False)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
