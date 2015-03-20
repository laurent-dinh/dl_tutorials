from theano import tensor


def sequence_map(fn, input_state, mask=None):
    batch_size = input_state.shape[1]
    n_timesteps = input_state.shape[0]
    input_dim = input_state.shape[2]

    if mask is not None:
        idx_mask = mask.nonzero()
        input_state = input_state[idx_mask[0], idx_mask[1]]
    else:
        input_state = input_state.reshape((n_timesteps * batch_size,
                                           input_dim))

    output_state_flat = fn(input_state)
    output_dim = output_state_flat.shape[1]
    output_state = tensor.zeros((n_timesteps, batch_size, output_dim))

    if mask is not None:
        output_state = tensor.inc_subtensor(output_state[idx_mask[0],
                                                         idx_mask[1]],
                                            output_state_flat)
        output_state = (output_state, mask)
    else:
        output_state = output_state_flat.reshape(
            (n_timesteps, batch_size, output_dim)
        )

    return output_state
