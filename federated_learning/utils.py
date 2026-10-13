"""Model construction and parameter exchange for the paper's dense FedAvg runs."""

from collections import OrderedDict

import numpy as np
import torch

from models.AquaCast import Model


def set_seed(seed):
    # Preserve FederatedMonarch's driver seeding.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_server_model(args):
    model = Model(args).float()
    # The legacy AquaCast path skipped CoLA replacement and used one Adam
    # parameter group per tensor, with all learning-rate multipliers equal to 1.
    groups = [
        {'params': [p], 'lr': args.lr,
         'weight_decay': args.weight_decay if p.dim() >= 2 and p.requires_grad else 0.0}
        for p in model.parameters()
    ]
    optimizer = torch.optim.Adam(groups, lr=args.lr, fused=args.device == 'cuda')
    return model, optimizer


def get_parameters(model):
    return [value.detach().cpu().numpy() for value in model.state_dict().values()]


def set_parameters(model, parameters):
    reference = model.state_dict()
    if len(parameters) != len(reference):
        raise ValueError('Received parameters do not match the AquaCast model.')
    state = OrderedDict(
        (name, torch.from_numpy(value).to(dtype=ref.dtype))
        for (name, ref), value in zip(reference.items(), parameters)
    )
    model.load_state_dict(state, strict=True)
