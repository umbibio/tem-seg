import warnings
import tensorflow as tf
from tensorflow.keras.backend import epsilon
from keras.saving import register_keras_serializable

@register_keras_serializable()
class MyWeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, pos_weight: float = 2., name: str = 'weighted_binary_crossentropy', from_logits: bool = False, **kwargs) -> None:
        # map legacy "auto" → new default
        if kwargs.get("reduction") == "auto":
            kwargs["reduction"] = "sum_over_batch_size"

        super().__init__(name=name, **kwargs)
        self.pos_weight = float(pos_weight)
        self.from_logits = bool(from_logits)

    def call(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)
        from_logits = self.from_logits

        # Use logits whenever they are available. `softmax` and `sigmoid`
        # activations cache logits on the `y_pred` Tensor.
        if hasattr(y_pred, '_keras_logits'):
            y_pred = y_pred._keras_logits  # pylint: disable=protected-access
            if from_logits:
                warnings.warn(
                    '"`binary_crossentropy` received `from_logits=True`, but the `y_pred`'
                    ' argument was produced by a sigmoid or softmax activation and thus '
                    'does not represent logits. Was this intended?"',
                    stacklevel=2)
            from_logits = True

        if from_logits:
            return tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=self.pos_weight)

        if (not isinstance(y_pred, (tf.__internal__.EagerTensor, tf.Variable)) and
            y_pred.op.type == 'Sigmoid') and not hasattr(y_pred, '_keras_history'):
            # When sigmoid activation function is used for y_pred operation, we
            # use logits from the sigmoid function directly to compute loss in order
            # to prevent collapsing zero when training.
            assert len(y_pred.op.inputs) == 1
            y_pred = y_pred.op.inputs[0]
            return tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=self.pos_weight)

        epsilon_ = tf.constant(epsilon(), dtype=y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

        # Compute cross entropy from probabilities.
        wbce = y_true * tf.math.log(y_pred + epsilon()) * self.pos_weight
        wbce += (1 - y_true) * tf.math.log(1 - y_pred + epsilon())
        return -wbce

    def get_config(self):
        config = {
            'from_logits': self.from_logits,
            'pos_weight': self.pos_weight,
        }
        base_config = super(MyWeightedBinaryCrossEntropy, self).get_config()

        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
