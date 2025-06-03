from .losses import MyWeightedBinaryCrossEntropy
from .metrics import MyMeanDSC, MyMeanIoU, MyJaccardIndex, MyF1Score, MyF2Score
from keras.layers import SpatialDropout2D as _BaseSpatialDropout2D
from keras.layers import Conv2DTranspose as _BaseConv2DTranspose


class SpatialDropout2D(_BaseSpatialDropout2D):
    def __init__(self, *args, **kwargs):
        for key in ["trainable", "noise_shape"]:
            kwargs.pop(key, None)
        super().__init__(*args, **kwargs)


class Conv2DTranspose(_BaseConv2DTranspose):
    def __init__(self, *args, **kwargs):
        for key in ["groups"]:
            kwargs.pop(key, None)
        super().__init__(*args, **kwargs)


custom_objects = {
    'MyWeightedBinaryCrossEntropy': MyWeightedBinaryCrossEntropy,
    'MyMeanIoU': MyMeanIoU,
    'MyMeanDSC': MyMeanDSC,
    'MyJaccardIndex': MyJaccardIndex,
    'MyF1Score': MyF1Score,
    'MyF2Score': MyF2Score,
    'Conv2DTranspose': Conv2DTranspose,
    'SpatialDropout2D': SpatialDropout2D,
}
