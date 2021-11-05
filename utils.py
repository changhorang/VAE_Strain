import torch
import torch.nn as nn
import numpy as np

class vae_loss(nn.Module):
    def __init__(self, pad_idx=0, anneal_function='logistic', k=0.0025, x0=2500):
        """Initialize Loss for VAE model.
        
        Args:
            pad_idx (int): index of <pad> token --삭제?
            anneal_function (str): anneal function for KL Divergence
            k (float): kl_anneal_function parameter
            x0 (int): kl_anneal_function parameter
        """
        super(vae_loss, self).__init__()
        self.NLL = nn.NLLLoss(ignore_index=pad_idx, reduction='sum')
        self.anneal_function = anneal_function.lower()
        self.k = k
        self.x0 = x0

    def kl_anneal_function(self, step):
        """Anneal kl_weight based on step
        Args:
            step (int): current training step1
        Return:
            kl_weight (float): annealed kl_weight
        """
        if self.anneal_function == 'logistic':
            return float(1 / (1 + np.exp(-self.k*(step - self.x0))))
        elif self.anneal_function == 'linear':
            return min(1, step/self.x0)
        elif self.anneal_function == 'none':
            return 1
        else:
            raise NotImplementedError("Only ['logistic', 'linear', 'none'] are supported for anneal_function")
        
    def forward(self, log_prob, target, length, mean, logvar, step):
        """Compute NLL loss and KL Divergence loss for VAE model
        Args:
            log_prob (torch.Tensor): log_prob of predicted token
            target (torch.Tensor): target token
            length (torch.Tensor): non-pad length of target token
            mean (torch.Tensor): mean of latent vector
            logvar (torch.Tensor): log variance of latent vector
            step (int): current training step
        Returns:
            NLL_loss (torch.Tensor): Negative Log Likelihood Loss
            KL_loss (torch.Tensor): Kullback–Leibler Divergence loss
            KL_weight (float): annealed kl_weight
        """
        log_prob = log_prob[:, :torch.max(length).item()].contiguous().view(-1, log_prob.size(2)) # log_prob: (batch_size * max_len, vocab_size)
        target = target[:, :torch.max(length).item()].contiguous().view(-1) # target: (batch_size * max_len)
        
        NLL_loss = self.NLL(log_prob, target)
        KL_div = -0.5*torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        KL_weight = self.kl_anneal_function(step)

        return NLL_loss, KL_div, KL_weight