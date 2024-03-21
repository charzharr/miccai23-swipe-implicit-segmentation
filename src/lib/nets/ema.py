"""
Tools to create and update EMA models.
"""


def create_ema_model(model):
    """ Given a newly made network, detach all its parameters so its 
    parameters can serve as the EMA of another model. """
    for param in model.parameters():
        param.detach_()
    return model


def update_ema_model(model, ema_model, alpha, iteration_num=None):
    """ ema_param_t = alpha * param_t-1 + (1 - alpha) * param_t
    Args:
        model: model with gradient updates
        ema_model: net with parameters that are an EMA of model
        alpha: fractional weight put on the pre-update param value
    """
    if iteration_num is not None:
        # Use the true average until the exponential average is more correct
        # iter=0 alpha=0, iter=1 alpha=0.5, iter=2 alpha=0.67, ...
        alpha = min(1 - 1 / (iteration_num + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data = alpha * ema_param.data + (1 - alpha) * param.data