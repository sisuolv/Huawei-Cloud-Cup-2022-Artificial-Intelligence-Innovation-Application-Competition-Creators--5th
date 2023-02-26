
import collections
import torch
from model_hunliu_small import MultiModal


model_lists=['./save/model_hunliu_small_best_2021.pt',
             './save/model_hunliu_small_best_2022.pt',
            
            ]
models = []
for model_list in model_lists:
    
    
    model = MultiModal('./xxxx/', './medbert/')   
    model.load_state_dict(torch.load(model_list, map_location='cpu'))
    model.eval()
    models.append(model)


worker_state_dict = [x.state_dict() for x in models]
weight_keys = list(worker_state_dict[0].keys())
fed_state_dict = collections.OrderedDict()
for key in weight_keys:
    key_sum = 0
    for i in range(len(models)):
        key_sum = key_sum + worker_state_dict[i][key]
    fed_state_dict[key] = key_sum / len(models)
#### update fed weights to fl model
model.load_state_dict(fed_state_dict)

torch.save(model, './model_hunliu_small_best_all.pt' )