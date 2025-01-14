import torch
import torch.nn.functional as F


def one_hot_encoding(Sxx_voice, Sxx_noise):
    output = torch.zeros(Sxx_voice.shape + (2,))
    output[..., 0] = (Sxx_voice > Sxx_noise).float()
    output[..., 1] = (Sxx_voice < Sxx_noise).float()
    return output
    
def Trans(tensor):
    return tensor.permute(0, 2, 1)

def norm(tensor):
    batch_size = tensor.size()[0]
    tensor_sq = torch.mul(tensor, tensor)
    tensor_sq = tensor_sq.view(batch_size, -1)
    return torch.sqrt(torch.sum(tensor_sq, dim=1))

def loss_dc(output, one_hot_encoding, Sxx_mag_mix):
    """
    Arguments:
    output: (batch_size, time_dim, frequency_dim, embedding_dim)
    one_hot_encoding: (batch_size, time_dim, frequency_dim, 2)
    Sxx_mag_mix: (batch_size, time_dim, frequency_dim)

    We have adapted our code from https://github.com/nussl/nussl/blob/master/nussl/ml/train/loss.py
    """
    embedding = output
    B,T,F,D = output.shape
    embedding = embedding.view(B, -1, D)
    Sxx_mag_mix = Sxx_mag_mix.detach().view(B, -1)
    one_hot_encoding = one_hot_encoding.view(B, -1, 2)

    # remove the loss of silence TF regions
    silence_mask = one_hot_encoding.sum(2, keepdim=True)
    embedding = silence_mask * embedding

    # referred as weight WR
    # W_i = |x_i| / \sigma_j{|x_j|}
    weights = torch.sqrt(Sxx_mag_mix / Sxx_mag_mix.sum(1, keepdim=True))
    one_hot_encoding = one_hot_encoding * weights.view(B, T*F, 1)
    embedding = embedding * weights.view(B, T*F, 1)

    # do batch affinity matrix computation
    loss_est = norm(torch.bmm(Trans(embedding), embedding))
    loss_est_true = 2*norm(torch.bmm(Trans(embedding), one_hot_encoding))
    loss_true = norm(torch.bmm(Trans(one_hot_encoding), one_hot_encoding))
    loss_embedding = loss_est - loss_est_true + loss_true

    return (loss_embedding * Sxx_mag_mix.sum(1, keepdim=True)).mean()