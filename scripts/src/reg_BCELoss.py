import torch

class reg_BCELoss(torch.nn.BCELoss):                                            
    def __init__(self, *args, epsilon=1e-6, dim=None, **kwargs):                
        torch.nn.BCELoss.__init__(self, *args, **kwargs)                        
        assert isinstance(dim, int)                                             
        self.regularization_dim = dim                                           
        self.regularization_A = epsilon                                         
        self.regularization_B = 1. - epsilon*self.regularization_dim

    def forward(self, inputs, target, *args, **kwargs):                         
        assert inputs.shape[-1]==self.regularization_dim                        
        transformed_input = self.regularization_A + self.regularization_B*inputs

        return torch.nn.BCELoss.forward(self, transformed_input, target, *args, **kwargs)
