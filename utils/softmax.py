from theano import tensor


def softmax(x):
    max_score = tensor.shape_padright(x.max(axis=-1))
    log_denominator = tensor.log(tensor.exp(x - max_score).sum(axis=-1))
    rval = tensor.exp(x - max_score - tensor.shape_padright(log_denominator))

    return rval
