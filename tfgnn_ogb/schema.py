"""
Created by building a `GraphTensor` with `data.generate_graph_samples` and
inspecting with `.spec`. The shapes of the features then needed to be updated
to support an unknown number of nodes and edges e.g. (None, 128). In practice,
these would be serialized as Protocol Buffers.

Node features seemingly need to be called `hidden_state` to work correctly
with the `tfgnn.keras.ConvGNNBuilder` layer. There is some discussion in the
Issue here: https://github.com/tensorflow/gnn/issues/5.

I also found that GraphTensors needed a context to work with
`merge_batch_to_components`. I was getting an exception about an IndexError
relating to self._data[_GraphPieceWithFeatures._DATAKEY_FEATURES].
"""
import tensorflow as tf
import tensorflow_gnn as tfgnn

TYPE_SPECS = {
    "ogbn-arxiv": tfgnn.GraphTensorSpec(
        {
            "context": tfgnn.ContextSpec(
                {
                    "features": {},
                    "sizes": tf.TensorSpec(
                        shape=(1,), dtype=tf.int32, name=None
                    ),
                },
                tf.TensorShape([]),
                tf.int32,
                None,
            ),
            "node_sets": {
                "paper": tfgnn.NodeSetSpec(
                    {
                        "features": {
                            "hidden_state": tf.TensorSpec(
                                shape=(None, 128), dtype=tf.float32, name=None
                            ),
                            "label": tf.TensorSpec(
                                shape=(None, 1), dtype=tf.int64, name=None
                            ),
                        },
                        "sizes": tf.TensorSpec(
                            shape=(1,), dtype=tf.int32, name=None
                        ),
                    },
                    tf.TensorShape([]),
                    tf.int32,
                    None,
                )
            },
            "edge_sets": {
                "cites": tfgnn.EdgeSetSpec(
                    {
                        "features": {},
                        "sizes": tf.TensorSpec(
                            shape=(1,), dtype=tf.int32, name=None
                        ),
                        "adjacency": tfgnn.AdjacencySpec(
                            {
                                "#index.0": tf.TensorSpec(
                                    shape=(None,), dtype=tf.int64, name=None
                                ),
                                "#index.1": tf.TensorSpec(
                                    shape=(None,), dtype=tf.int64, name=None
                                ),
                            },
                            tf.TensorShape([]),
                            tf.int32,
                            {"#index.0": "paper", "#index.1": "paper"},
                        ),
                    },
                    tf.TensorShape([]),
                    tf.int32,
                    None,
                )
            },
        },
        tf.TensorShape([]),
        tf.int32,
        None,
    )
}
