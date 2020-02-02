import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Implements the algorithm presented in
    'Neural Discrete Representation Learning' by van den Oord et al.
    https://arxiv.org/abs/1711.00937
    Input any tensor to be quantized. Last dimension will be used as space in
    which to quantize. All other dimensions will be flattened and will be seen
    as different examples to quantize.
    The output tensor will have the same shape as the input.
    For example a tensor with shape [16, 32, 32, 64] will be reshaped into
    [16384, 64] and all 16384 vectors (each of 64 dimensions)  will be quantized
    independently.
    Args:
        embedding_dim: integer representing the dimensionality of the tensors in the
            quantized space. Inputs to the modules must be in this format as well.
        num_embeddings: integer, the number of vectors in the quantized space.
            commitment_cost: scalar which controls the weighting of the loss terms
            (see equation 4 in the paper - this variable is Beta).
    """

    def __init__(self, num_embeddings, embedding_dim,
                 commitment_cost, device, verbose=False, debug=False):
        super(VectorQuantizer, self).__init__()
        self.verbose = verbose
        self.debug = debug

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(
            self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(
            -1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost
        self._device = device

    def forward(self, inputs):
        if self.debug:
            print('shape of inputs in VectorQuantizer.forward', inputs.size())
            sys.stdout.flush()
        # convert inputs from BCHW (here its B, F, T) -> BHWC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        if self.debug:
            print(
                'shape of permutated inputs in VectorQuantizer.forward',
                input_shape)
            sys.stdout.flush()

        _, time, batch_size = input_shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        if self.debug:
            print(
                'shape of flat_input in VectorQuantizer.forward',
                flat_input.size())
            print('device of flat_input', flat_input.device)
            sys.stdout.flush()

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0],
            self._num_embeddings,
            device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(
            encodings, self._embedding.weight).view(input_shape)
        if self.debug:
            print(
                'shape of quantized in VectorQuantizer.forward',
                quantized.size())
            sys.stdout.flush()

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs *
                                          torch.log(avg_probs + 1e-10)))
        if self.debug:
            print(
                'shape of inputs quantized in VectorQuantizer.forward',
                quantized.size())
            print("with inputs quantized: ", quantized.size())
            sys.stdout.flush()

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(
            0, 2, 1).contiguous(), perplexity, encodings
