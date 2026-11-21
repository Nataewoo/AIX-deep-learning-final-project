import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn


def network(task, size, width):
    layers = [nn.Linear(size, width), nn.ReLU(), nn.Dropout(0.1),
              nn.Linear(width, width // 2), nn.ReLU(), nn.Linear(width // 2, 1)]
    layers.append(nn.Softplus() if task == "A" else nn.Sigmoid())
    return nn.Sequential(*layers)


def restore(task, values):
    return values * 30 if task == "A" else 1 + values * 4


def fit(task, params, inputs, targets, seeds, monitor_inputs=None, validation_score=None):
    scaler = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
    )
    features_tensor = torch.tensor(scaler.fit_transform(inputs), dtype=torch.float32)
    y_scaled = targets / 30 if task == "A" else (targets - 1) / 4
    target_tensor = torch.tensor(y_scaled, dtype=torch.float32)
    loss_fn = nn.L1Loss() if task == "A" else nn.MSELoss()
    models, history = [], []
    if monitor_inputs is not None:
        monitor_x = torch.tensor(scaler.transform(monitor_inputs), dtype=torch.float32)
    for seed in seeds:
        torch.manual_seed(seed)
        net = network(task, inputs.shape[1], params["width"])
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
        for epoch in range(1, params["epochs"] + 1):
            net.train()
            for batch in torch.randperm(len(targets)).split(32):
                optimizer.zero_grad()
                loss = loss_fn(net(features_tensor[batch]).ravel(), target_tensor[batch])
                loss.backward()
                optimizer.step()
            if monitor_inputs is not None:
                net.eval()
                with torch.inference_mode():
                    validation = restore(task, net(monitor_x).ravel().numpy())
                    history.append({
                        "seed": seed, "epoch": epoch,
                        "train_loss_scaled": float(loss_fn(net(features_tensor).ravel(), target_tensor)),
                        "validation_macro_mae": validation_score(validation),
                    })
        models.append(net.eval())
    return (scaler, models), history


def predict(task, fitted, inputs):
    scaler, models = fitted
    x = torch.tensor(scaler.transform(inputs), dtype=torch.float32)
    with torch.inference_mode():
        return restore(task, np.mean([net(x).ravel().numpy() for net in models], axis=0))
