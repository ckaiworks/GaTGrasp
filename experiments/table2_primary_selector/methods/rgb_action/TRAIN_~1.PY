import argparse
import csv
import importlib.util
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
HERE = Path(__file__).resolve().parent
BASE = load_module('explicit_lrc_base_v4b', str(Path(__file__).resolve().with_name('train_stage1_strictsuccess_explicitLRC_v4b.py')))

def parse_args():
    p = argparse.ArgumentParser(description='Old12 strict-success weighted BCE plus full-object annealed selection loss. Each selection group contains every candidate of one object.')
    p.add_argument('--train_csv', required=True, type=Path)
    p.add_argument('--val_csv', type=Path, default=None)
    p.add_argument('--test_csv', type=Path, default=None)
    p.add_argument('--run_dir', required=True, type=Path)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.0003)
    p.add_argument('--weight_decay', type=float, default=0.0001)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--no_gs', action='store_true')
    p.add_argument('--soft_top1_weight', type=float, default=0.1)
    p.add_argument('--select_temperature_start', type=float, default=1.0)
    p.add_argument('--select_temperature_end', type=float, default=0.1)
    p.add_argument('--soft_top1_warmup_epochs', type=int, default=5)
    p.add_argument('--fixed_epoch', type=int, default=0, help='0: use validation selection; >0: save exactly this epoch.')
    return p.parse_args()

def forward(model, b):
    return model(b['rgb'], b['action_geom'])

def group_indices(dataset):
    result = defaultdict(list)
    for (i, row) in enumerate(dataset.rows):
        result[int(row['object_id'])].append(i)
    result = {obj: idx for (obj, idx) in result.items()}
    bad = []
    for (obj, idx) in result.items():
        ys = [BASE.strict_label(dataset.rows[i]) for i in idx]
        if not (any((y == 1 for y in ys)) and any((y == 0 for y in ys))):
            bad.append((obj, len(idx), int(sum(ys))))
    if bad:
        raise RuntimeError(f'full-object ranking requires both labels per training object; bad={bad}')
    return result

def make_group_batch(dataset, indices, device):
    raw = [dataset[i] for i in indices]
    return BASE.move(default_collate(raw), device)

def annealed_select_loss(logits, labels, temperature):
    """Negative log probability that Top-1 belongs to the positive set."""
    group_logits = logits.reshape(-1)
    group_labels = labels.reshape(-1)

    positive_logits = group_logits[group_labels > 0.5]

    if positive_logits.numel() == 0:
        return group_logits.sum() * 0.0

    return -(
        torch.logsumexp(positive_logits / temperature, dim=0)
        - torch.logsumexp(group_logits / temperature, dim=0)
    )


def cosine_select_temperature(epoch, epochs, start, end):
    """Cosine anneal from a smooth listwise objective to a sharp Top-1 objective."""
    if start <= 0.0 or end <= 0.0:
        raise RuntimeError('selection temperatures must be positive')
    if start < end:
        raise RuntimeError('select_temperature_start must be >= select_temperature_end')
    progress = 0.0 if epochs <= 1 else (epoch - 1) / (epochs - 1)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def full_object_softtop1_score(model, dataset, groups, device, temperature):
    model.eval()
    values = []
    for indices in groups.values():
        b = make_group_batch(dataset, indices, device)
        logits = forward(model, b)
        mass = torch.softmax(logits / temperature, dim=0)[b['label'] > 0.5].sum()
        values.append(float(mass.item()))
    return float(np.mean(values)) if values else float('nan')

def main():
    args = parse_args()
    if args.fixed_epoch > 0 and args.val_csv is not None:
        raise RuntimeError('--fixed_epoch is for final train/test mode; omit --val_csv')
    if args.fixed_epoch == 0 and args.val_csv is None:
        raise RuntimeError('selection mode requires --val_csv')
    if args.select_temperature_start <= 0.0 or args.select_temperature_end <= 0.0:
        raise RuntimeError('selection temperatures must be positive')
    if args.select_temperature_start < args.select_temperature_end:
        raise RuntimeError('select_temperature_start must be >= select_temperature_end')
    if args.run_dir.exists():
        raise RuntimeError(f'refuse existing run_dir: {args.run_dir}')
    args.run_dir.mkdir(parents=True)
    BASE.set_seed(args.seed)
    stats_path = args.run_dir / 'fullobject_feature_stats.npz'
    BASE.prepare_stats(args.train_csv, stats_path)
    train_set = BASE.LRCDataset(args.train_csv, stats_path, args.image_size)
    val_set = BASE.LRCDataset(args.val_csv, stats_path, args.image_size) if args.val_csv else None
    test_set = BASE.LRCDataset(args.test_csv, stats_path, args.image_size) if args.test_csv else None
    train_groups = group_indices(train_set)
    val_groups = group_indices(val_set) if val_set else None
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args) if val_set else None
    test_loader = DataLoader(test_set, shuffle=False, **loader_args) if test_set else None
    pos = sum((BASE.strict_label(r) == 1 for r in train_set.rows))
    neg = len(train_set.rows) - pos
    if pos == 0 or neg == 0:
        raise RuntimeError(f'invalid train P/N={pos}/{neg}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = BASE.ExplicitLRCModel(no_gs=args.no_gs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print('device:', device, 'no_gs:', args.no_gs)
    print('strict P/N:', pos, neg, 'pos_weight:', float(pos_weight))
    print('full-object groups:', len(train_groups), 'sizes:', {str(k): len(v) for (k, v) in sorted(train_groups.items())})
    print('annealed_select:', {'weight': args.soft_top1_weight, 'temperature_start': args.select_temperature_start, 'temperature_end': args.select_temperature_end, 'schedule': 'cosine', 'warmup_epochs': args.soft_top1_warmup_epochs})
    best = -float('inf')
    best_path = args.run_dir / 'best_fullobject_softtop1.pt'
    log_path = args.run_dir / 'train_log.csv'
    for epoch in range(1, args.epochs + 1):
        model.train()
        (bce_total, bce_count) = (0.0, 0)
        for batch in train_loader:
            b = BASE.move(batch, device)
            logit = forward(model, b)
            loss = criterion(logit, b['label'])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            bce_total += float(loss.item()) * len(logit)
            bce_count += len(logit)
        warm = min(1.0, epoch / max(args.soft_top1_warmup_epochs, 1))
        lam = args.soft_top1_weight * warm
        select_temperature = cosine_select_temperature(
            epoch,
            args.epochs,
            args.select_temperature_start,
            args.select_temperature_end,
        )
        soft_values = []
        object_order = list(train_groups)
        random.shuffle(object_order)
        for obj in object_order:
            b = make_group_batch(train_set, train_groups[obj], device)
            logit = forward(model, b)
            soft = annealed_select_loss(logit, b['label'], select_temperature)
            weighted_soft = lam / len(object_order) * soft
            opt.zero_grad(set_to_none=True)
            weighted_soft.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            soft_values.append(float(soft.item()))
        sch.step()
        row = {'epoch': epoch, 'lr': opt.param_groups[0]['lr'], 'train_bce': bce_total / max(bce_count, 1), 'train_annealed_select_loss': float(np.mean(soft_values)), 'select_temperature': select_temperature, 'soft_top1_weight_effective': lam}
        if val_set is not None:
            val = BASE.evaluate(model, val_loader, device, criterion)
            val_soft = full_object_softtop1_score(model, val_set, val_groups, device, args.select_temperature_end)
            score = val['mean_ap']
            row.update({'val_soft_top1': val_soft, 'val_map': val['mean_ap'], 'val_top1': val['top1_success'], 'val_top5': val['top5_success'], 'val_auroc': val['auroc'], 'selection_score': score})
            if score > best:
                best = float(score)
                torch.save({'model': model.state_dict(), 'epoch': epoch, 'best_score': best, 'selection_metric': 'val_mean_ap'}, best_path)
        else:
            score = -float(epoch)
            if epoch == args.fixed_epoch:
                best = float(epoch)
                torch.save({'model': model.state_dict(), 'epoch': epoch, 'best_score': best, 'selection_metric': 'fixed_epoch_from_nested_validation'}, best_path)
        with log_path.open('a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if epoch == 1:
                w.writeheader()
            w.writerow(row)
        if val_set is not None:
            print(f"[Epoch {epoch:03d}/{args.epochs}] bce={row['train_bce']:.4f} selectL={row['train_annealed_select_loss']:.4f} T={select_temperature:.4f} w={lam:.4f} val_softTop1={row['val_soft_top1']:.4f} val_map={row['val_map']:.4f} val_top1={row['val_top1']:.4f} select={row['selection_score']:.4f} best={best:.4f}")
        else:
            print(f"[Epoch {epoch:03d}/{args.epochs}] bce={row['train_bce']:.4f} selectL={row['train_annealed_select_loss']:.4f} T={select_temperature:.4f} w={lam:.4f}")
    if not best_path.is_file():
        raise RuntimeError('no checkpoint was saved')
    state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(state['model'])
    best_info = {'epoch': int(state['epoch']), 'selection_metric': state['selection_metric'], 'best_score': float(state['best_score']), 'used_outer_test_for_selection': False}
    (args.run_dir / 'best_val.json').write_text(json.dumps(best_info, indent=2))
    if test_loader is not None:
        test = BASE.evaluate(model, test_loader, device, criterion)
        test_groups = group_indices(test_set)
        test['soft_top1_success'] = full_object_softtop1_score(model, test_set, test_groups, device, args.select_temperature_end)
        (args.run_dir / 'test_metrics.json').write_text(json.dumps(test, indent=2))
        print('\n===== TEST =====')
        print(json.dumps(test, indent=2))
    print('DONE:', args.run_dir)
if __name__ == '__main__':
    main()
