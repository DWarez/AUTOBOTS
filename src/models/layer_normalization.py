import torch
import torch.nn as nn



class LayerNormalization(nn.Module):
    """Layer Normalization class"""
    def __init__(self, d_model, epsilon=1e-10):
        """LayerNormalization constructor

        Args:
            d_model (int):  shape of the of the input tensor or the last dimension of the input tensor
            epsilon (float, optional): parameter of variance. Defaults to 1e-10.
        """
        super(LayerNormalization, self).__init__()
        if isinstance(d_model, int):
            d_model = (d_model,)
        else:
            d_model = (d_model[-1],)
        self.d_model = torch.Size(d_model)

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.epsilon = epsilon


    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        return y * self.gamma + self.beta