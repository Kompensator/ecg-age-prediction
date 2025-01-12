import json
import torch
import os
from tqdm import tqdm
from resnet import ResNet1d
from dataloader import BatchDataloader
from torch.utils.data import DataLoader
import sys
sys.path.append(r"C:\Users\dingyi.zhang\Documents\DeepLearningECG\dataset")
from CLSA_dataset import CLSA
import torch.optim as optim
import numpy as np


def compute_loss(ages, pred_ages, weights):         # this looks like a version of MSE loss
    diff = ages.flatten() - pred_ages.flatten()
    loss = torch.sum(weights.flatten() * diff * diff)
    return loss


def train(ep, dataload):
    model.train()
    total_loss = 0
    n_entries = 0
    train_desc = "Epoch {:2d}: train - Loss: {:.6f}"
    train_bar = tqdm(initial=0, leave=True, total=len(dataload),
                     desc=train_desc.format(ep, 0, 0), position=0)
    for batch in dataload:
        traces = batch[0]
        targets = batch[args.pred_target]
        weights = batch[7]
        ages = batch[6]
        sexes = batch[9]
        tabular = torch.stack((ages, sexes), 1)

        traces, targets, weights, tabular = traces.to(device), targets.to(device), weights.to(device), tabular.to(device)
        # Reinitialize grad
        model.zero_grad()
        # Send to device
        # Forward pass
        pred_ages = model(traces)
        loss = compute_loss(targets, pred_ages, weights)
        # Backward pass
        loss.backward()
        # Optimize
        optimizer.step()
        # Update
        bs = len(traces)
        total_loss += loss.detach().cpu().numpy()
        n_entries += bs
        # Update train bar
        train_bar.desc = train_desc.format(ep, total_loss / n_entries)
        train_bar.update(1)
    train_bar.close()
    return total_loss / n_entries


def eval(ep, dataload):
    model.eval()
    total_loss = 0
    n_entries = 0
    eval_desc = "Epoch {:2d}: valid - Loss: {:.6f}"
    eval_bar = tqdm(initial=0, leave=True, total=len(dataload),
                    desc=eval_desc.format(ep, 0, 0), position=0)
    for batch in dataload:
        traces = batch[0]
        targets = batch[args.pred_target]
        weights = batch[7]
        ages = batch[6]
        sexes = batch[9]
        tabular = torch.stack((ages, sexes), 1)

        traces, targets, weights, tabular = traces.to(device), targets.to(device), weights.to(device), tabular.to(device)
        with torch.no_grad():
            # Forward pass
            pred_ages = model(traces)
            loss = compute_loss(targets, pred_ages, weights)
            # Update outputs
            bs = len(traces)
            # Update ids
            total_loss += loss.detach().cpu().numpy()
            n_entries += bs
            # Print result
            eval_bar.desc = eval_desc.format(ep, total_loss / n_entries)
            eval_bar.update(1)
    eval_bar.close()
    return total_loss / n_entries


if __name__ == "__main__":
    import pandas as pd
    import argparse
    from warnings import warn

    # Arguments that will be saved in config file
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Train model to predict rage from the raw ecg tracing.')
    parser.add_argument('--epochs', type=int, default=140,
                        help='maximum number of epochs (default: 140)')
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed for number generator (default: 2)')
    parser.add_argument('--sample_freq', type=int, default=400,
                        help='sample frequency (in Hz) in which all traces will be resampled at (default: 400)')
    parser.add_argument('--seq_length', type=int, default=4096,
                        help='size (in # of samples) for all traces. If needed traces will be zeropadded'
                                    'to fit into the given size. (default: 4096)')
    parser.add_argument('--scale_multiplier', type=int, default=10,
                        help='multiplicative factor used to rescale inputs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size (default: 32).')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument("--patience", type=int, default=7,
                        help='maximum number of epochs without reducing the learning rate (default: 7)')
    parser.add_argument("--min_lr", type=float, default=1e-7,
                        help='minimum learning rate (default: 1e-7)')
    parser.add_argument("--lr_factor", type=float, default=0.1,
                        help='reducing factor for the lr in a plateu (default: 0.1)')
    parser.add_argument('--net_filter_size', type=int, nargs='+', default=[64, 128, 196, 256, 320],
                        help='filter size in resnet layers (default: [64, 128, 196, 256, 320]).')
    parser.add_argument('--net_seq_lengh', type=int, nargs='+', default=[4096, 1024, 256, 64, 16],
                        help='number of samples per resnet layer (default: [4096, 1024, 256, 64, 16]).')
    parser.add_argument('--dropout_rate', type=float, default=0.8,
                        help='dropout rate (default: 0.8).')
    parser.add_argument('--kernel_size', type=int, default=17,
                        help='kernel size in convolutional layers (default: 17).')
    parser.add_argument('--folder', default='model/',
                        help='output folder (default: ./out)')
    parser.add_argument('--traces_dset', default='tracings',
                        help='traces dataset in the hdf5 file.')
    parser.add_argument('--ids_dset', default='',
                        help='by default consider the ids are just the order')
    parser.add_argument('--age_col', default='age',
                        help='column with the age in csv file.')
    parser.add_argument('--ids_col', default=None,
                        help='column with the ids in csv file.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='use cuda for computations. (default: False)')
    parser.add_argument('--n_valid', type=int, default=100,
                        help='the first `n_valid` exams in the hdf will be for validation.'
                             'The rest is for training')
    parser.add_argument('--pred_target', type=int, default=1,
                        help='Index of the torch dataloader output to use for target.'
                        '1 for FI52 (default), 5 for BPM, 6 for age, 8 for FI39'
                        'MUST CHANGE THE WEIGHT CALCULATION IN DATASET TOO!')
    parser.add_argument('--run_name', type=str, default='default')

    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    torch.manual_seed(args.seed)
    # Set device
    device = torch.device(args.device)

    tqdm.write("Building data loaders...")
    train_dataset = CLSA(start=0, end=25000)
    valid_dataset = CLSA(start=25000, end=27500)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    N_LEADS = 12  # the 12 leads
    N_CLASSES = 1  # just the age
    model = ResNet1d(input_dim=(N_LEADS, args.seq_length),
                     blocks_dim=list(zip(args.net_filter_size, args.net_seq_lengh)),
                     n_classes=N_CLASSES,
                     kernel_size=args.kernel_size,
                     dropout_rate=args.dropout_rate,
                     mlp_output=0)
    model.load_state_dict(torch.load(r'C:\Users\dingyi.zhang\Documents\ecg-age-prediction\checkpoints\CLSA_age_pred.pth')['model'])
    model.to(device=device)

    optimizer = optim.Adam(model.parameters(), args.lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience,
                                                     min_lr=args.lr_factor * args.min_lr,
                                                     factor=args.lr_factor)

    tqdm.write("Training...")
    start_epoch = 0
    best_loss = np.Inf
    history = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss', 'lr',
                                    'weighted_rmse', 'weighted_mae', 'rmse', 'mse'])
    for ep in range(start_epoch, args.epochs):
        train_loss = train(ep, train_loader)
        valid_loss = eval(ep, valid_loader)
        # Save best model
        if valid_loss < best_loss:
            # Save model
            torch.save({'epoch': ep,
                        'model': model.state_dict(),
                        'valid_loss': valid_loss,
                        'optimizer': optimizer.state_dict()},
                       os.path.join(r'C:\Users\dingyi.zhang\Documents\ecg-age-prediction\checkpoints', args.run_name + '.pth'))
            # Update best validation loss
            best_loss = valid_loss
        # Get learning rate
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
        # Interrupt for minimum learning rate
        if learning_rate < args.min_lr:
            break
        # Print message
        tqdm.write('Epoch {:2d}: \tTrain Loss {:.6f} ' \
                  '\tValid Loss {:.6f} \tLearning Rate {:.7f}\t'
                 .format(ep, train_loss, valid_loss, learning_rate))
        # Save history
        history = history.append({"epoch": ep, "train_loss": train_loss,
                                  "valid_loss": valid_loss, "lr": learning_rate}, ignore_index=True)
        history.to_csv(os.path.join(r"C:\Users\dingyi.zhang\Documents\ecg-age-prediction", 'history.csv'), index=False)
        # Update learning rate
        scheduler.step(valid_loss)
    tqdm.write("Done!")