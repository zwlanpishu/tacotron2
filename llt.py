import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def reverse_padded_sequence(inputs, lengths, batch_first=False):
    """ Reverses sequences according to their lengths.
    Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
    ``B x T x *`` if True. T is the length of the longest sequence
    (or larger), B is the batch size, and * is any number of dimensions
    (including 0).

    Arguments:
        inputs (Variable): padded batch of variable length sequences.
        lengths (list[int]): list of sequence lengths
        batch_first (bool, optional): if True, inputs should be B x T x *.
    Returns:
        A Variable with the same size as inputs, but with each sequence
        reversed according to its length.
    """

    if batch_first:
        inputs = inputs.transpose(0, 1)
    max_length, batch_size = inputs.size(0), inputs.size(1)

    if len(lengths) != batch_size:
        raise ValueError("inputs is incompatible with lengths.")

    ind = [
        list(reversed(range(0, length))) + list(range(length, max_length))
        for length in lengths
    ]
    ind = Variable(torch.LongTensor(ind).transpose(0, 1))

    for dim in range(2, inputs.dim()):
        ind = ind.unsqueeze(dim)
    ind = ind.expand_as(inputs)

    if inputs.is_cuda:
        ind = ind.cuda(inputs.get_device())

    reversed_inputs = torch.gather(inputs, 0, ind)
    if batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs


padded_sequence = torch.Tensor(
    [
        [[1, 4], [2, 4], [3, 2], [4, 1], [1, 1]],
        [[1, 3], [2, 1], [3, 2], [0, 0], [0, 0]],
        [[1, 5], [2, 1], [0, 0], [0, 0], [0, 0]],
    ]
)


length = torch.LongTensor([5, 3, 2])
length = length.numpy().tolist()
print(padded_sequence.size())
print(padded_sequence)

lstm = nn.LSTM(2, 2, batch_first=True)
input_packed = pack_padded_sequence(padded_sequence, length, batch_first=True)
encoder_outputs_packed, (h_last, c_last) = lstm(input_packed)
encoder_outputs, _ = pad_packed_sequence(
    encoder_outputs_packed, batch_first=True
)
print(encoder_outputs)


# r = reverse_padded_sequence(padded_sequence, length, batch_first=True)
# print(r)
# print(r.size())
