

def get_data_components(config):
    """ Called by run_experiment.py to load data components. 
    Job is to route data components factory to the correct dataset functions.
    """
    data_config = config.data 
    if data_config.name == 'polyp_sessile':
        from .data.polyp_sessile.api import get_dataset_components
    elif data_config.name == 'bcv':
        from .data.bcv.api import get_dataset_components
    else:
        raise NotImplementedError()
    
    return get_dataset_components(config)

