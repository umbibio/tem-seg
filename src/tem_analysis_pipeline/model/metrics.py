import tensorflow as tf
# from tensorflow_addons.metrics import F1Score, FBetaScore
from keras import Metric
from tensorflow.keras.backend import epsilon
from keras.saving import register_keras_serializable


@register_keras_serializable()
class MyMeanIoU(Metric):
    def __init__(self, name='mean_iou', from_logits=False, threshold=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.iou = self.add_weight(name='iou', initializer='zeros')
        self.cnt = self.add_weight(name='cnt', initializer='zeros')
        self.from_logits = from_logits
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.convert_to_tensor(y_true, dtype=self.dtype)
        y_pred = tf.convert_to_tensor(y_pred, dtype=self.dtype)

        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)

        if self.threshold is not None:
            y_pred = y_pred > self.threshold
            y_pred = tf.cast(y_pred, dtype=self.dtype)

        reduce_axis = list(range(1, len(y_pred.shape)))
        intersection = tf.reduce_sum(y_pred * y_true, axis=reduce_axis)
        union = tf.reduce_sum(y_pred + y_true, axis=reduce_axis) - intersection + epsilon()
        values = intersection / union

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            values = tf.multiply(values, sample_weight)
            count =  tf.reduce_sum(sample_weight)
        else:
            count = tf.cast(len(values), self.dtype)

        value = tf.reduce_sum(values)
        self.iou.assign_add(value)
        self.cnt.assign_add(count)

    def result(self):
        return self.iou / self.cnt

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.iou.assign(0.)
        self.cnt.assign(0.)
    
    def get_config(self):
        config = {'from_logits': self.from_logits}
        base_config = super().get_config()

        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@register_keras_serializable()
class MyMeanDSC(Metric):
    def __init__(self, name='mean_dsc', from_logits=False, threshold=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dsc = self.add_weight(name='dsc', initializer='zeros')
        self.cnt = self.add_weight(name='cnt', initializer='zeros')
        self.from_logits = from_logits
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.convert_to_tensor(y_true, dtype=self.dtype)
        y_pred = tf.convert_to_tensor(y_pred, dtype=self.dtype)

        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)

        if self.threshold is not None:
            y_pred = y_pred > self.threshold
            y_pred = tf.cast(y_pred, dtype=self.dtype)

        reduce_axis = list(range(1, len(y_pred.shape)))
        numerator = 2 * tf.reduce_sum(y_pred * y_true, axis=reduce_axis)
        denominator = tf.reduce_sum(y_pred, axis=reduce_axis) + tf.reduce_sum(y_true, axis=reduce_axis) + epsilon()
        values = numerator / denominator

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            values = tf.multiply(values, sample_weight)
            count =  tf.reduce_sum(sample_weight)
        else:
            count = tf.cast(len(values), self.dtype)

        value = tf.reduce_sum(values)
        self.dsc.assign_add(value)
        self.cnt.assign_add(count)

    def result(self):
        return self.dsc / self.cnt

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.dsc.assign(0.)
        self.cnt.assign(0.)

    def get_config(self):
        config = {'from_logits': self.from_logits}
        base_config = super().get_config()

        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@register_keras_serializable()
class MyConfussionMatrixBaseClass(Metric):
    def __init__(self, name='confussion_matrix', from_logits=False, threshold=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.from_logits = from_logits
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.convert_to_tensor(y_true, dtype=self.dtype)
        y_pred = tf.convert_to_tensor(y_pred, dtype=self.dtype)

        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)

        if self.threshold is not None:
            y_pred = y_pred > self.threshold
            y_pred = tf.cast(y_pred, dtype=self.dtype)
        
        reduce_axis = list(range(1, len(y_pred.shape)))
        tp = tf.reduce_sum(y_pred * y_true, axis=reduce_axis)
        fp = tf.reduce_sum(y_pred * (1 - y_true), axis=reduce_axis)
        fn = tf.reduce_sum((1 - y_pred) * y_true, axis=reduce_axis)
        tn = tf.reduce_sum((1 - y_pred) * (1 - y_true), axis=reduce_axis)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            tp = tf.multiply(tp, sample_weight)
            fp = tf.multiply(fp, sample_weight)
            fn = tf.multiply(fn, sample_weight)
            tn = tf.multiply(tn, sample_weight)

        self.tp.assign_add(tf.reduce_sum(tp))
        self.fp.assign_add(tf.reduce_sum(fp))
        self.fn.assign_add(tf.reduce_sum(fn))
        self.tn.assign_add(tf.reduce_sum(tn))

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.tp.assign(0.)
        self.fp.assign(0.)
        self.fn.assign(0.)
        self.tn.assign(0.)

    def get_config(self):
        config = {'from_logits': self.from_logits, 'threshold': self.threshold}
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@register_keras_serializable()
class MyJaccardIndex(MyConfussionMatrixBaseClass):
    def __init__(self, name='jaccard_index', **kwargs):
        super().__init__(name=name, **kwargs)
    
    def result(self):
        numerator = self.tp
        denominator = self.tp + self.fp + self.fn + epsilon()
        return numerator / denominator

@register_keras_serializable()
class MyF1Score(MyConfussionMatrixBaseClass):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
    
    def result(self):
        numerator = 2 * self.tp
        denominator = numerator + self.fn + self.fp + epsilon()
        return numerator / denominator

@register_keras_serializable()
class MyF2Score(MyConfussionMatrixBaseClass):
    def __init__(self, name='f2_score', **kwargs):
        super().__init__(name=name, **kwargs)
    
    def result(self):
        numerator = 5 * self.tp
        denominator = numerator +  4 * self.fn + self.fp + epsilon()
        return numerator / denominator


if False:
    import numpy as np
    import tensorflow_addons as tfa
    from metrics import *

    shape = (100, 10, 10, 1)
    a, b = 5, 2
    y_true = np.random.randint(2, size=shape, dtype=int)
    y_pred = np.empty(shape, dtype=float)
    y_pred[y_true == 1] = np.random.beta(a, b, size=(y_true == 1).sum())
    y_pred[y_true == 0] = np.random.beta(b, a, size=(y_true == 0).sum())
    accuracy = ((y_true - (y_pred > 0.5).astype(int)) == 0).sum() / np.prod(shape)
    accuracy
    sample_weight = np.random.randint(1, 10, size=shape[0], dtype=int)

    metric = MyMeanIoU(threshold=0.5)
    metric.update_state(y_true, y_pred)
    metric.result().numpy()

    metric = MyMeanDSC(threshold=0.5)
    metric.update_state(y_true, y_pred)
    metric.result().numpy()

    metric = MyJaccardIndex(threshold=0.5)
    metric.update_state(y_true, y_pred)
    metric.result().numpy()

    metric = MyF1Score(threshold=0.5)
    metric.update_state(y_true, y_pred)
    metric.result().numpy()

    metric = MyF2Score(threshold=0.5)
    metric.update_state(y_true, y_pred)
    metric.result().numpy()

    metric = tfa.metrics.F1Score(num_classes=1, threshold=0.5, average='micro')
    metric.update_state(y_true, y_pred)
    metric.result().numpy()

    metric = F1Score(threshold=0.5, average='micro')
    metric.update_state(y_true, y_pred)
    metric.result().numpy()

    metric = F2Score(threshold=0.5, average='micro')
    metric.update_state(y_true, y_pred)
    metric.result().numpy()

