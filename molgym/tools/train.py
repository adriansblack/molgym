import logging
import time
from typing import Dict, Any, Tuple, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import CheckpointHandler, CheckpointState
from .torch_tools import to_numpy, dict_to_device
from .utils import ProgressLogger


def train(
    model: torch.nn.Module,
    loss_fn: Callable,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR,
    start_epoch: int,
    max_num_epochs: int,
    patience: int,
    checkpoint_handler: CheckpointHandler,
    logger: ProgressLogger,
    eval_interval: int,
    device: torch.device,
):
    lowest_loss = np.inf
    patience_counter = 0

    logging.info('Started training')
    for epoch in range(start_epoch, max_num_epochs):
        # Train
        for batch in train_loader:
            _, opt_metrics = take_step(model=model, loss_fn=loss_fn, batch=batch, optimizer=optimizer, device=device)
            opt_metrics['mode'] = 'opt'
            opt_metrics['epoch'] = epoch
            logger.log(opt_metrics)

        # Validate
        if epoch % eval_interval == 0:
            valid_loss, eval_metrics = evaluate(model=model, loss_fn=loss_fn, data_loader=valid_loader, device=device)
            eval_metrics['mode'] = 'eval'
            eval_metrics['epoch'] = epoch
            logger.log(eval_metrics)

            logging.info(f'Epoch {epoch}: loss={valid_loss:.4f}')

            if valid_loss >= lowest_loss:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f'Stopping optimization after {patience_counter} epochs without improvement')
                    break
            else:
                lowest_loss = valid_loss
                patience_counter = 0
                checkpoint_handler.save(state=CheckpointState(model, optimizer, lr_scheduler), epochs=epoch)

        # LR scheduler
        lr_scheduler.step()

    logging.info('Training complete')


def take_step(
    model: torch.nn.Module,
    loss_fn: Callable,
    batch: Dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    start_time = time.time()
    batch = dict_to_device(batch, device)
    optimizer.zero_grad()
    output, _aux = model(batch['state'], batch['action'], training=True)
    loss = loss_fn(pred=output, ref=batch)
    loss.backward()
    optimizer.step()

    loss_dict = {
        'loss': to_numpy(loss),
        'time': time.time() - start_time,
    }

    return loss, loss_dict


def evaluate(
    model: torch.nn.Module,
    loss_fn: Callable,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    total_loss = 0.0

    start_time = time.time()
    for batch in data_loader:
        batch = dict_to_device(batch, device)
        output, aux = model(batch['state'], batch['action'], training=False)
        batch = dict_to_device(batch, torch.device('cpu'))
        output = dict_to_device(output, torch.device('cpu'))

        loss = loss_fn(pred=output, ref=batch)
        total_loss += to_numpy(loss).item()

    avg_loss = total_loss / len(data_loader)

    aux = {
        'loss': avg_loss,
        'time': time.time() - start_time,
    }

    return avg_loss, aux
