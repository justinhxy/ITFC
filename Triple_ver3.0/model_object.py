from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from Net.TripleNetwork import *
from Net.api import *
from loss_function import joint_loss

models = {
    'Triple':{
        'Name': 'Triple_model_CrossAttentionFusion',
        'Model': Triple_model_CrossAttentionFusion,
        'Loss': CrossEntropyLoss,
        'Optimizer': Adam,
        'batch_size': 8,
        'Lr': 0.0001,
        'Epoch': 200,
        'Run': run

    },
'Triple+KAN+self':{
        'Name': 'Triple_model_CrossAttentionFusion',
        'Model': Triple_model_CrossAttentionFusion_self_KAN,
        'Loss': joint_loss,
        'Optimizer': Adam,
        'batch_size': 8,
        'Lr': 0.0001,
        'Epoch': 200,
        'w1': 0.2,
        'w2': 0.01,
        'Run': run_main_1

    }
}