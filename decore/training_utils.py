import torch
from tqdm import tqdm


def validate(model, val_loader, device):
    model.eval()
    total_samples = 0
    correct_pred = 0
    accuracy = None

    with torch.no_grad():
        with tqdm(val_loader, desc='Val') as pbar:
            for i, (X, y) in enumerate(pbar):
                X = X.float().to(device)
                y = y.long().to(device)

                outputs = model(X)
                _, y_pred = torch.max(outputs.data, 1)

                correct_pred += (y_pred == y).sum().item()
                total_samples += X.size(0)
                accuracy = correct_pred / total_samples
                pbar.set_postfix(acc=accuracy)
    return accuracy


def train(model, agents, train_loader, optimizer_model, optimizer_agents,
          criterion_model, penalty, device, optimize_agents=True,
          optimize_model=True):
    if optimize_model is True:
        model.train()
    else:
        model.eval()

    if optimize_agents is True:
        for agent in agents:
            agent.reset()
            agent.train()
    else:
        for agent in agents:
            agent.reset()
            agent.eval()

    total_samples = 0
    correct_pred = 0
    total_channels = sum(agent.num_subagents for agent in agents)
    logs = {
        'total_samples': 0,
        'num_channels_dropped': 0,
        'total_channels': total_channels,
        'accuracy': 0,
        'loss_model_avg': 0,
        'loss_agents_avg': 0,
        'accuracy_reward_avg': 0,
        'compression_reward_avg': 0,
        'full_reward_avg': 0,
        'weights_avg': 0,
        'probabilities_avg': 0,
        'p_min': 0,
        'p_max': 0
    }

    with tqdm(train_loader, desc='Train') as pbar:
        for i, (X, y) in enumerate(pbar):
            X = X.float().to(device)
            y = y.long().to(device)

            # Model optimization

            if optimize_model is True:
                optimizer_model.zero_grad()
                outputs = model(X)
                loss_model = criterion_model(outputs, y)
                loss_model.backward()
                optimizer_model.step()
            else:
                outputs = model(X)
                loss_model = criterion_model(outputs, y)

            y_pred = torch.argmax(outputs, dim=1)
            correct_pred += (y_pred == y).sum().item()
            total_samples += X.size(0)
            accuracy = correct_pred / total_samples

            # Agent optimization

            dropped = 0  # Number of dropped channels
            w_avg = 0
            p_avg = 0
            p_min = 1
            p_max = 0
            loss_agents = 0

            optimizer_agents.zero_grad()
            for agent in agents:
                a = agent.a.detach()  # shape (n, c)
                p = agent.p  # shape (n, c)

                rc = (1 - a).sum((1,), keepdims=True)  # shape (n, 1)
                racc = torch.where(y_pred == y, 1, -penalty)  # shape (n,)
                racc = racc.unsqueeze(-1)  # shape (n, 1) 

                r = rc * racc
                r = r.detach()

                l = torch.log(a * p + (1 - a) * (1 - p)) * r
                loss_agents += l.mean((1,), keepdims=True)  # shape (n, 1)

                # Logs
                dropped += (1 - a).sum().item() / X.size(0)
                w_avg += agent.w.mean().item() / len(agents)
                p_avg += p.mean().item() / len(agents)
                p_min = min(p_min, p.min().item())
                p_max = max(p_max, p.max().item())

            loss_agents = -loss_agents.mean()
            if optimize_agents is True:
                loss_agents.backward()
                optimizer_agents.step()

            logs['num_channels_dropped'] = dropped
            logs['accuracy'] = accuracy
            logs['loss_model_avg'] += loss_model.item()
            logs['loss_agents_avg'] += loss_agents.item()
            logs['accuracy_reward_avg'] += racc.sum().item()
            logs['compression_reward_avg'] += rc.sum().item()
            logs['full_reward_avg'] += r.sum().item()
            logs['weights_avg'] = w_avg
            logs['probabilities_avg'] = p_avg
            logs['p_min'] = p_min
            logs['p_max'] = p_max
            pbar.set_postfix(loss_a=logs['loss_agents_avg'] / total_samples,
                             loss_m=logs['loss_model_avg'] / total_samples,
                             acc=logs['accuracy'],
                             dropped=logs['num_channels_dropped'],
                             w=logs['weights_avg'],
                             p=logs['probabilities_avg'],
                             p_min=logs['p_min'],
                             p_max=logs['p_max'])

    logs['total_samples'] = total_samples
    logs['loss_model_avg'] = logs['loss_model_avg'] / total_samples
    logs['loss_agents_avg'] = logs['loss_agents_avg'] / total_samples
    logs['accuracy_reward_avg'] = logs['accuracy_reward_avg'] / total_samples
    logs['compression_reward_avg'] = logs['compression_reward_avg'] / total_samples
    logs['full_reward_avg'] = logs['full_reward_avg'] / total_samples
    return logs
