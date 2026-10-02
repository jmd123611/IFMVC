import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from dataloader import load_mfeat
from fine_tuning_losses import stage1_kl_loss, stage2_cross_entropy_loss
from loss import Loss
from metrics import evaluate
from network import Network
from post_tree import tree_fusion


LEARNING_RATE = 1e-4


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="IFMVC Mfeat reproduction")
    parser.add_argument("--data-root", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "results")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mse-epochs", type=int, default=200)
    parser.add_argument("--con-epochs", type=int, default=50)
    parser.add_argument("--tune-epochs", type=int, default=50)
    parser.add_argument("--final-tune-epochs", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--temperature-f", type=float, default=0.5)
    parser.add_argument("--temperature-l", type=float, default=1.0)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--high-feature-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                      generator=generator, num_workers=0)


def full_batch(dataset, data_size):
    return next(iter(DataLoader(dataset, batch_size=data_size, shuffle=False)))


def match_labels(reference, prediction, device):
    reference = np.asarray(reference, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    dimension = int(max(reference.max(), prediction.max()) + 1)
    weights = np.zeros((dimension, dimension), dtype=np.int64)
    for index in range(len(reference)):
        weights[prediction[index], reference[index]] += 1
    rows, cols = linear_sum_assignment(weights.max() - weights)
    matched = np.zeros(len(reference), dtype=np.int64)
    for row, col in zip(rows, cols):
        matched[reference == col] = row
    return torch.from_numpy(matched).long().to(device)


def pretrain_epoch(model, loader, optimizer, device, view_count):
    model.train()
    mse = torch.nn.MSELoss()
    for views, _, _ in loader:
        views = [view.to(device) for view in views]
        optimizer.zero_grad()
        _, _, reconstructions, _ = model(views)
        loss = sum(mse(views[index], reconstructions[index]) for index in range(view_count))
        loss.backward()
        optimizer.step()


def contrastive_epoch(model, loader, optimizer, criterion, device, view_count):
    model.train()
    mse = torch.nn.MSELoss()
    for views, _, _ in loader:
        views = [view.to(device) for view in views]
        optimizer.zero_grad()
        features, assignments, reconstructions, _ = model(views)
        losses = []
        for first in range(view_count):
            for second in range(first + 1, view_count):
                losses.append(criterion.forward_feature(features[first], features[second]))
                losses.append(criterion.forward_label(assignments[first], assignments[second]))
            losses.append(mse(views[first], reconstructions[first]))
        sum(losses).backward()
        optimizer.step()


def make_pseudo_labels(model, dataset, data_size, view_count, class_count, device, seed):
    views, _, _ = full_batch(dataset, data_size)
    views = [view.to(device) for view in views]
    model.eval()
    with torch.no_grad():
        features, _, _, _ = model(views)
    scaler = MinMaxScaler()
    labels = []
    for view_index in range(view_count):
        representation = scaler.fit_transform(features[view_index].cpu().numpy())
        labels.append(KMeans(n_clusters=class_count, n_init=100, random_state=seed)
                      .fit_predict(representation))
    return labels


def stage1_epoch(model, dataset, data_size, pseudo_labels, optimizer, device):
    views, _, _ = full_batch(dataset, data_size)
    views = [view.to(device) for view in views]
    optimizer.zero_grad()
    _, assignments, _, _ = model(views)
    losses = []
    for view_index, pseudo_label in enumerate(pseudo_labels):
        prediction = torch.argmax(assignments[view_index].detach(), dim=1).cpu().numpy()
        target = match_labels(pseudo_label, prediction, device)
        losses.append(stage1_kl_loss(assignments[view_index], target))
    sum(losses).backward()
    optimizer.step()


def deterministic_vote(label_matrix):
    class_count = int(label_matrix.max()) + 1
    return np.asarray([
        np.argmax(np.bincount(row, minlength=class_count)) for row in label_matrix
    ], dtype=int)


def stage2_epoch(model, dataset, data_size, pseudo_labels, optimizer, device):
    views, _, _ = full_batch(dataset, data_size)
    views = [view.to(device) for view in views]
    optimizer.zero_grad()
    _, assignments, _, _ = model(views)
    aligned = []
    for view_index, pseudo_label in enumerate(pseudo_labels):
        prediction = torch.argmax(assignments[view_index].detach(), dim=1).cpu().numpy()
        aligned.append(match_labels(pseudo_label, prediction, device).cpu().numpy())
    consensus = torch.from_numpy(deterministic_vote(np.stack(aligned, axis=1))).long().to(device)
    sum(stage2_cross_entropy_loss(assignment, consensus) for assignment in assignments).backward()
    optimizer.step()


def infer(model, dataset, data_size, device):
    views, truth, _ = full_batch(dataset, data_size)
    device_views = [view.to(device) for view in views]
    model.eval()
    with torch.no_grad():
        _, predictions = model.forward_cluster(device_views)
    partitions = np.stack([prediction.cpu().numpy() for prediction in predictions], axis=1)
    return partitions.astype(int), np.asarray(truth).astype(int), [view.numpy() for view in views]


def run_seed(args, seed):
    set_seed(seed)
    device = torch.device(args.device)
    dataset, dimensions, view_count, data_size, class_count = load_mfeat(args.data_root)
    batch_size = min(args.batch_size, data_size)
    loader = make_loader(dataset, batch_size, seed)
    model = Network(view_count, dimensions, args.feature_dim, args.high_feature_dim,
                    class_count, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=args.weight_decay)
    criterion = Loss(batch_size, class_count, args.temperature_f,
                     args.temperature_l, device).to(device)
    print(f"Seed {seed}: optimizer learning rate = {optimizer.param_groups[0]['lr']}")

    start = time.perf_counter()
    for _ in range(args.mse_epochs):
        pretrain_epoch(model, loader, optimizer, device, view_count)
    for _ in range(args.con_epochs):
        contrastive_epoch(model, loader, optimizer, criterion, device, view_count)
    pseudo_labels = make_pseudo_labels(model, dataset, data_size, view_count,
                                       class_count, device, seed)
    for _ in range(args.tune_epochs):
        stage1_epoch(model, dataset, data_size, pseudo_labels, optimizer, device)
    pseudo_labels = make_pseudo_labels(model, dataset, data_size, view_count,
                                       class_count, device, seed)
    for _ in range(args.final_tune_epochs):
        stage2_epoch(model, dataset, data_size, pseudo_labels, optimizer, device)

    partitions, truth, views = infer(model, dataset, data_size, device)
    tree, prediction, centers = tree_fusion(partitions, views, class_count)
    result = {"seed": seed, **evaluate(truth, prediction),
              "elapsed_seconds": time.perf_counter() - start,
              "learning_rate": LEARNING_RATE,
              "algorithm2_center": "coordinate-wise median"}

    seed_dir = args.output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    np.save(seed_dir / "view_specific_partitions.npy", partitions)
    np.save(seed_dir / "algorithm2_median_centers.npy", centers)
    np.save(seed_dir / "tree_predictions.npy", prediction)
    (seed_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    print(f"Device: {args.device}")
    print(f"Learning rate: {LEARNING_RATE}")
    results = [run_seed(args, seed) for seed in args.seeds]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
