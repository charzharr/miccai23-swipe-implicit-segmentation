
import torch
from fvcore.nn import FlopCountAnalysis


@torch.no_grad()
def get_model_flops(model, input_shape, device=None, disp=False):
    """ Returns the total number of MAdds of the model given an input shape 
    It returns MAdds not FLOPS, I KNOW BUT THATS WHATS REPORTED IN ML PAPERS.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    input = torch.ones(()).new_empty(input_shape, 
                                     dtype=next(model.parameters()).dtype,
                                     device=device)
    flops = FlopCountAnalysis(model, input)
    model_flops = flops.total()
    
    if disp:
        print(f"MAdds: {round(model_flops * 1e-9, 2)} G")
    return model_flops
    
