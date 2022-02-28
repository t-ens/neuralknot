from tensorflow import cast, shape, ones
from tensorflow.keras.backend import ctc_batch_cost

def ctc_loss(y, y_p):
    batch_len = cast(shape(y)[0], dtype="int64")
    input_length = cast(shape(y_p)[1], dtype="int64")
    label_length = cast(shape(y)[1], dtype="int64")

    input_length = input_length * ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * ones(shape=(batch_len, 1), dtype="int64")

    loss = ctc_batch_cost(y, y_p, input_length, label_length)
    return loss     


class CTCGRU(GaussEncoder):
    """
        Similar to SimpleGRU just using connectionist temporal classification
        rather than categorical cross-entropy. Ultimately a modification of
        CTC to account for the special structure of Gauss codes is planned this
        is just a testing model. 
    """
    pass
