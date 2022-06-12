import math
import torch
import torch.nn as nn


class ScaleDotProductAttention(nn.Module):
    """Scale Dot Product Attention mechanism"""
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                    value: torch.Tensor, mask: torch.Tensor=None, e=1e-12):
        """ScaleDotProductAttention forward step

        Args:
            query (torch.Tensor): Q as for query
            key (torch.Tensor): K as for key
            value (torch.Tensor): V as for value
            mask (torch.Tensor, optional): mask to apply to the scores. 
                Defaults to None.
            e (float, optional): value to set masked elements. 
                Defaults to 1e-12.

        Returns:
            tuple(torch.Tensor, torch.Tensor): attention result, softmax output
        """
        d_k = query.size(-1)
        scores = (query @ key.transpose(2, 3))/math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask==0, -e)
        
        p_attention = self.softmax(scores)

        return torch.matmul(p_attention, value), p_attention

    
class MultiHeadAttention(nn.Module):
    """Multi Head Attention mechanism"""
    def __init__(self, d_model: int, n_heads: int, 
                    attention=ScaleDotProductAttention()) -> None:
        """MultiHeadAttention constructor

        Args:
            d_model (int): shape of the of the input tensor or the last 
                dimension of the input tensor
            n_heads (int): number of heads attention units
            attention (nn.Module, optional): type of attention mechanism. 
                Defaults to ScaleDotProductAttention.
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model if isinstance(d_model, int) else d_model[-1]
        self.n_heads = n_heads
        self.attention = attention

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        # Linear layer for concatenation
        self.linear_c = nn.Linear(d_model, d_model)

        
    def split(self, _tensor: torch.Tensor) -> torch.Tensor:
        """Splits the original tensor by the number of heads

        Args:
            _tensor (torch.tensor): tensor to be splitted 
                [batch_size, length, d_model]
        Returns:
            torch.tensor: tensor with shape [batch_size, head, length, d_v]
        """
        batch_size, length, d_model = _tensor.size()

        assert d_model % self.n_heads == 0, "d_model must be divisible with \
                                                the number of heads"
        d_v = d_model//self.n_heads

        _tensor = _tensor.view(batch_size, length, self.n_heads, d_v)\
                                                            .transpose(1, 2)
        return _tensor


    def concatenate(self, _tensor: torch.Tensor) -> torch.Tensor:
        """Groups the tensor splits after the attention mechanism is applied

        Args:
            tensor (torch.tensor): tensor with shape 
                [batch_size, head, length, d_v]
        Returns:
            torch.tensor: tensor with shape [batch_size, length, d_model]
        """
        batch_size, n_heads, length, d_v = _tensor.size()
        d_model = n_heads * d_v
        return _tensor.transpose(1, 2).contiguous()\
                                            .view(batch_size, length, d_model)


    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        """Forward step of MHA

        Args:
            query (torch.Tensor): Q as for query
            key (torch.Tensor): K as for key
            value (torch.Tensor): V as for value
            mask (torch.Tensor, optional): mask to apply during attention. 
                Defaults to None.

        Returns:
            torch.Tensor: output of Linear layer
        """
        # Pass Q, K, V through linear layers
        h_q, h_k, h_v = self.linear_q(query), self.linear_k(key), \
                                                        self.linear_v(value)
        
        # Split tensors by number of heads
        h_q, h_k, h_v = self.split(h_q), self.split(h_k), self.split(h_v)

        # Compute attention for the splitted, hidden representations of Q, K, V
        result, p_attention = self.attention(h_q, h_k, h_v, mask=mask)

        # Concatenate the result and pass it through the linear layer
        result = self.linear_c(self.concatenate(result))

        return result