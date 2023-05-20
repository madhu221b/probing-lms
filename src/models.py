import torch.nn as nn
import torch


class StructuralProbe(nn.Module):
    """ Computes squared L2 distance after projection by a matrix.
    For a batch of sentences, computes all n^2 pairs of distances
    for each sentence in the batch.
    """
    def __init__(self, model_dim, rank, device="cpu"):
        super().__init__()
        self.probe_rank = rank
        self.model_dim = model_dim
        
        self.proj = nn.Parameter(data = torch.zeros(self.model_dim, self.probe_rank))
        
        nn.init.uniform_(self.proj, -0.05, 0.05)
        self.to(device)

    def forward(self, batch):
        """ Computes all n^2 pairs of distances after projection
        for each sentence in a batch.
        Note that due to padding, some distances will be non-zero for pads.
        Computes (B(h_i-h_j))^T(B(h_i-h_j)) for all i,j
        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of distances of shape (batch_size, max_seq_len, max_seq_len)
        """
        transformed = torch.matmul(batch, self.proj)
        batchlen, seqlen, rank = transformed.size()
        
        transformed = transformed.unsqueeze(2)
        transformed = transformed.expand(-1, -1, seqlen, -1)
        transposed = transformed.transpose(1,2)
        
        diffs = transformed - transposed
        
        squared_diffs = diffs.pow(2)
        squared_distances = torch.sum(squared_diffs, -1)
        
        return squared_distances


class PolynomialProbe(nn.Module):
    def __init__(self, model_dim, rank, device="cpu"):
        super().__init__()
        self.probe_rank = rank
        self.model_dim = model_dim
        self.c = 1
        self.d = 4
        self.proj = nn.Parameter(data = torch.zeros(self.model_dim, self.probe_rank))
        
        nn.init.uniform_(self.proj, -0.05, 0.05)
        self.to(device)

    def forward(self, batch):
        """ Computes all n^2 pairs of distances after projection
        for each sentence in a batch.
        Note that due to padding, some distances will be non-zero for pads.
        Computes ((Bh_i)^T(Bh_j) + c)^d for all i,j
        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of distances of shape (batch_size, max_seq_len, max_seq_len)
        """
        transformed = torch.matmul(batch, self.proj) # B*h
        batchlen, seqlen, rank = transformed.size()
        
        mulmatrix = torch.bmm(transformed.view(batchlen ,seqlen, rank), # (Bh_i)^T(Bh_j)
        transformed.view(batchlen, rank, seqlen))
        mulmatrix = (mulmatrix+self.c)
        polydist = torch.pow(mulmatrix,self.d)
        return polydist


class RbfProbe(nn.Module):
    def __init__(self, model_dim, rank, device="cpu"):
        super().__init__()
        self.probe_rank = rank
        self.model_dim = model_dim
        self.sigma = 1
        self.proj = nn.Parameter(data = torch.zeros(self.model_dim, self.probe_rank))
        
        nn.init.uniform_(self.proj, -0.05, 0.05)
        self.to(device)

    def forward(self, batch):
        """ Computes all n^2 pairs of distances after projection
        for each sentence in a batch.
        Note that due to padding, some distances will be non-zero for pads.
        Computes exp(-||Bh_i - Bh_j||^2 / 2sigma^2) for all i,j
        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of distances of shape (batch_size, max_seq_len, max_seq_len)
            """
        transformed = torch.matmul(batch, self.proj)
        batchlen, seqlen, rank = transformed.size()
        transformed = transformed.unsqueeze(2)
        transformed = transformed.expand(-1, -1, seqlen, -1)
        transposed = transformed.transpose(1,2)
        diffs = transformed - transposed
        squared_diffs = diffs.pow(2)
        squared_distances = torch.sum(squared_diffs, -1)
        exp_dists = torch.exp(-squared_distances/(2*pow(self.sigma,2)))
        return exp_dists

class SigmoidProbe(nn.Module):
    def __init__(self, model_dim, rank, device="cpu"):
        super().__init__()
        self.probe_rank = rank
        self.model_dim = model_dim
        self.a = 1
        self.b = 0
        self.proj = nn.Parameter(data = torch.zeros(self.model_dim, self.probe_rank))
        
        nn.init.uniform_(self.proj, -0.05, 0.05)
        self.to(device)

    def forward(self, batch):
        """ Computes all n^2 pairs of distances after projection
        for each sentence in a batch.
        Note that due to padding, some distances will be non-zero for pads.
        Computes tanh(a(Bh_i)^T(Bh_j) + b) for all i,j
        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of distances of shape (batch_size, max_seq_len, max_seq_len)
            """
        transformed = torch.matmul(batch, self.proj)
        batchlen, seqlen, rank = transformed.size()
        transformed = torch.matmul(batch, self.proj) # B*h
        batchlen, seqlen, rank = transformed.size()
        
        mulmatrix = torch.bmm(transformed.view(batchlen ,seqlen, rank), # (Bh_i)^T(Bh_j)
        transformed.view(batchlen, rank, seqlen))
        tanh_dists = torch.tanh(self.a*mulmatrix + self.b)
        return tanh_dists

