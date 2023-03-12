import torch


class Agent:
    def __init__(self, w, prob_threshold=0.5,
                 threshold_type='BINARY'):
        self.w = w  # weights
        self.p = None  # probabilities
        self.a = None  # actions

        self.prob_threshold = prob_threshold
        self.threshold_type = threshold_type  # BINARY and BINARY_INV
        self.num_subagents = self.w.size(dim=0)
        self.is_eval = False
        self.tracker = {}  # To track forward passes through the attached
        # modules. Need this for preserving the agent's state when multiple
        # modules share the same agent.
        self.handles = {}  # Handles to detach the agents from the modules.
        self.masks = {}  # Masks for pruning channels.

    def attach_module(self, module, module_name):
        def hook(module, input, output):
            batch_size = output.size(dim=0)
            num_dims_to_insert = output.ndim - 2

            a = self.get_actions(module_name, n=batch_size)  # shape (N, C)
            for i in range(num_dims_to_insert):
                a = a.unsqueeze(-1)
            return output * a
        
        if module_name in self.tracker:
            raise Exception('Module already attached to the agent.')

        handle = module.register_forward_hook(hook)
        self.handles[module_name] = handle
        self.tracker[module_name] = False
        self.masks[module_name] = {'weight': torch.zeros_like(module.weight.data)}

    def get_actions(self, module_name, n=1):
        if all(x is True for x in self.tracker.values()) or \
                all(x is False for x in self.tracker.values()):
            # Reset agent's state: probabilities, actions, and tracker.
            self.reset()
            
            # Compute the new probabilities.
            p = torch.sigmoid(self.w)
            p = p.unsqueeze(0)
            repeats = (n,) + (1,) * (p.ndim - 1)
            p = p.repeat(repeats)
            # p = p.repeat_interleave(repeats=n, dim=0)  # No MPS support.
            self.p = p

            # If eval, actions are computed by thresholding probabilities.
            # If train, actions are computed using bernouille sampling.
            if self.is_eval:
                if self.threshold_type == 'BINARY':
                    a = torch.where(p > self.prob_threshold, 1, 0)
                elif self.threshold_type == 'BINARY_INV':
                    a = torch.where(p < self.prob_threshold, 1, 0)
                else:
                    raise NotImplementedError
            else:
                a = torch.bernoulli(p)
            self.a = a.detach()
        else:
            if self.a is None:
                raise Exception('Reset agents.')            
        self.tracker[module_name] = True  # Forward pass done for this module.
        return self.a

    def eval(self, prob_threshold=0.5, threshold_type='BINARY'):
        self.change_prob_threshold(prob_threshold, threshold_type)
        self.is_eval = True
        self.w.requires_grad = False

    def train(self):
        self.is_eval = False
        self.w.requires_grad = True

    def change_prob_threshold(self, prob_threshold, threshold_type='BINARY'):
        self.threshold_type = threshold_type
        self.prob_threshold = prob_threshold

    def reset(self):
        self.p = None
        self.a = None
        # self.w.grad = None
        self.tracker.update((k, False) for k in self.tracker)
    
    def get_masks(self):
        for module_name in self.masks:
            a = self.get_actions(module_name=module_name, n=1)
            a = a.squeeze()
            self.masks[module_name]['weight'].fill_(0)
            self.masks[module_name]['weight'][torch.nonzero(a)] = 1
        return self.masks       
    

def get_num_features(module):
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv1d)):
        return module.out_channels
    elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
        return module.num_features
    elif isinstance(module, torch.nn.Linear):
        return module.out_features
    else:
        raise NotImplementedError


def attach_agents(model, modules_to_prune, device='cpu', init_weight=6.9):
    name_to_module = {}
    agents = []
    name_to_agent = {}
    num_subagents = 0
    num_agents = len(modules_to_prune)

    for name, module in model.named_modules(remove_duplicate=False):
        name_to_module[name] = module
    
    for i, x in enumerate(modules_to_prune):
        # All modules inside a sublist share the same agent.
        if isinstance(x, str):
            module = name_to_module[x]
            num_features = get_num_features(module)

            w = torch.ones((num_features,), device=device) * init_weight
            w.requires_grad_(True)  # leaf tensor
            agent = Agent(w)
            agent.attach_module(module, x)

            agents.append(agent)
            num_subagents += agent.num_subagents
            name_to_agent[x] = agent

        elif isinstance(x, list):
            module = name_to_module[x[0]]
            num_features = get_num_features(module)

            w = torch.ones((num_features,), device=device) * init_weight
            w.requires_grad_(True)  # leaf tensor
            agent = Agent(w)

            agents.append(agent)
            num_subagents += agent.num_subagents
            for name in x:
                module = name_to_module[name]
                assert get_num_features(module) == agent.num_subagents
                agent.attach_module(module, name)                
                name_to_agent[name] = agent
        else:
            raise TypeError('Supports str and [str, ...] only!')
    return agents, name_to_agent, num_agents, num_subagents
