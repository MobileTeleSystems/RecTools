import typing as tp

OptimizerType = tp.Literal["adam", "adagrad", "adamw"]
QuantizerType = tp.Literal["rqvae", "rkmeans"]
LRScheduleType = tp.Literal["constant", "cosine", "linear"]
