import torch
from tqdm import tqdm


def average_model_with_ckpts(module, ckpts):
    state_dict = {k: v for k, v in module.state_dict().items()}
    state2count = {k: 1 for k in state_dict.keys()}
    for c in tqdm(ckpts):
        state_dict2 = torch.load(c, map_location="cpu")["state_dict"]
        for k, v in state_dict2.items():
            state_dict[k] += v
            state2count[k] += 1
    state_dict = {k: v / state2count[k] for k, v in state_dict.items()}
    module.load_state_dict(state_dict)
    return module
