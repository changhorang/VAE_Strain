import torch
import torch.nn as nn
import numpy as np

class vae_loss(nn.Module):
    def __init__(self, anneal_function='logistic', k=0.0025, x0=2500):
        """Initialize Loss for VAE model."""
        super(vae_loss, self).__init__()
        self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.CrossEntropyLoss()
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

    def forward(self, log_prob, target, mean, log_var, step):
        recon_loss = self.loss_fn(log_prob, target)
        KL_loss = -0.5*torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        KL_weight = self.kl_anneal_function(step)
        
        total_loss = recon_loss + KL_loss*KL_weight

        return total_loss