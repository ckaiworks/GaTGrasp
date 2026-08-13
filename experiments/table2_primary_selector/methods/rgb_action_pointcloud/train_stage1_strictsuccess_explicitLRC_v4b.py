import argparse
import csv
import importlib.util
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
COMMON = Path(__file__).resolve().parents[4] / 'src' / 'stage1' / 'final' / 'stage1_common.py'
spec = importlib.util.spec_from_file_location('stage1_common_pointcloud', COMMON)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def strict_label(row):
    return common.get_float(row, 'label_twofinger_strict', common.get_float(row, 'label', 0.0))

def read_rows(path):
    return common.read_csv(path)

def prepare_stats(train_csv, out_path):
    rows = read_rows(train_csv)
    ag = np.stack([common.build_action_geom(r) for r in rows], axis=0)
    sums = np.zeros((12,), dtype=np.float64)
    sqs = np.zeros((12,), dtype=np.float64)
    count = 0
    for row in rows:
        d = np.load(row['local_gs_path'], allow_pickle=False)
        for (feat_key, mask_key) in [('left_feat', 'left_mask'), ('right_feat', 'right_mask'), ('corridor_feat', 'corridor_mask')]:
            feat = d[feat_key].astype(np.float32)
            valid = d[mask_key].astype(np.float32) > 0.5
            take = feat[valid]
            sums += take.sum(axis=0)
            sqs += (take ** 2).sum(axis=0)
            count += len(take)
    mean = sums / max(count, 1)
    var = sqs / max(count, 1) - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12)) + 1e-06
    np.savez(out_path, ag_mean=ag.mean(axis=0).astype(np.float32), ag_std=(ag.std(axis=0) + 1e-06).astype(np.float32), lrc_mean=mean.astype(np.float32), lrc_std=std.astype(np.float32))
    print('[saved stats]', out_path)
    print('action/LRC dims:', ag.shape[1], mean.shape[0])

class _OriginalLRCDataset(Dataset):

    def __init__(self, csv_path, stats_path, image_size=224):
        self.rows = read_rows(csv_path)
        self.image_size = image_size
        s = np.load(stats_path)
        (self.ag_mean, self.ag_std) = (s['ag_mean'], s['ag_std'])
        (self.lrc_mean, self.lrc_std) = (s['lrc_mean'], s['lrc_std'])
        self.rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.rows)

    def rgb(self, path):
        im = Image.open(path).convert('RGB').resize((self.image_size, self.image_size), Image.BILINEAR)
        x = np.asarray(im).astype(np.float32) / 255.0
        x = (x - self.rgb_mean[None, None]) / self.rgb_std[None, None]
        return x.transpose(2, 0, 1).astype(np.float32)

    def patch(self, d, feat_key, mask_key):
        mask = d[mask_key].astype(np.float32)
        feat = d[feat_key].astype(np.float32)
        feat = (feat - self.lrc_mean[None]) / self.lrc_std[None]
        feat *= mask[:, None]
        return (feat.astype(np.float32), mask.astype(np.float32))

    def __getitem__(self, index):
        row = self.rows[index]
        d = np.load(row['local_gs_path'], allow_pickle=False)
        (left, left_mask) = self.patch(d, 'left_feat', 'left_mask')
        (right, right_mask) = self.patch(d, 'right_feat', 'right_mask')
        (corridor, corridor_mask) = self.patch(d, 'corridor_feat', 'corridor_mask')
        ag = (common.build_action_geom(row) - self.ag_mean) / self.ag_std
        return {'rgb': torch.from_numpy(self.rgb(row['vision_rgb_path'])), 'action_geom': torch.from_numpy(ag.astype(np.float32)), 'left': torch.from_numpy(left), 'right': torch.from_numpy(right), 'corridor': torch.from_numpy(corridor), 'left_mask': torch.from_numpy(left_mask), 'right_mask': torch.from_numpy(right_mask), 'corridor_mask': torch.from_numpy(corridor_mask), 'label': torch.tensor(strict_label(row), dtype=torch.float32), 'object_id': torch.tensor(int(row['object_id']), dtype=torch.long)}

class PatchEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(12, 64), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.out = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.15))

    def forward(self, feat, mask):
        (b, n, c) = feat.shape
        x = self.point(feat.reshape(b * n, c)).reshape(b, n, -1)
        m = mask.unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1.0)
        mean = (x * m).sum(dim=1) / denom
        mx = x.masked_fill(m < 0.5, -10000.0).max(dim=1).values
        return self.out(torch.cat([mean, mx], dim=1))

class _OriginalExplicitLRCModel(nn.Module):

    def __init__(self, no_gs=False):
        super().__init__()
        self.no_gs = no_gs
        self.rgb_encoder = common.RGBEncoder(out_dim=256)
        self.action_encoder = nn.Sequential(nn.Linear(13, 64), nn.ReLU(inplace=True), nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.endpoint_encoder = PatchEncoder()
        self.corridor_encoder = PatchEncoder()
        self.relation = nn.Sequential(nn.Linear(64 * 5, 128), nn.ReLU(inplace=True), nn.Dropout(0.25))
        self.head = nn.Sequential(nn.Linear(448, 256), nn.ReLU(inplace=True), nn.Dropout(0.25), nn.Linear(256, 1))

    def forward(self, rgb, action_geom, left, right, corridor, left_mask, right_mask, corridor_mask):
        if self.no_gs:
            left = torch.zeros_like(left)
            right = torch.zeros_like(right)
            corridor = torch.zeros_like(corridor)
            left_mask = torch.ones_like(left_mask)
            right_mask = torch.ones_like(right_mask)
            corridor_mask = torch.ones_like(corridor_mask)
        l = self.endpoint_encoder(left, left_mask)
        r = self.endpoint_encoder(right, right_mask)
        c = self.corridor_encoder(corridor, corridor_mask)
        gs = self.relation(torch.cat([l, r, c, torch.abs(l - r), l * r], dim=1))
        rgb = self.rgb_encoder(rgb)
        action = self.action_encoder(action_geom)
        return self.head(torch.cat([rgb, action, gs], dim=1)).squeeze(1)

def move(batch, device):
    return {k: v.to(device, non_blocking=True) for (k, v) in batch.items()}

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    (ys, ps, objs) = ([], [], [])
    (total, count) = (0.0, 0)
    for batch in loader:
        b = move(batch, device)
        logit = model(b['rgb'], b['action_geom'], b['pointcloud'], b['left'], b['right'], b['corridor'], b['left_mask'], b['right_mask'], b['corridor_mask'])
        loss = criterion(logit, b['label'])
        prob = torch.sigmoid(logit)
        total += float(loss.item()) * len(prob)
        count += len(prob)
        ys.extend(b['label'].cpu().numpy().tolist())
        ps.extend(prob.cpu().numpy().tolist())
        objs.extend(b['object_id'].cpu().numpy().tolist())
    out = common.safe_metrics(np.asarray(ys, dtype=np.int64), np.asarray(ps, dtype=np.float64), np.asarray(objs, dtype=np.int64))
    out['loss'] = total / max(count, 1)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split_root', required=True, type=Path)
    ap.add_argument('--run_dir', required=True, type=Path)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=0.0003)
    ap.add_argument('--weight_decay', type=float, default=0.0001)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--image_size', type=int, default=224)
    ap.add_argument('--no_gs', action='store_true')
    args = ap.parse_args()
    set_seed(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    stats_path = args.run_dir / 'lrc_feature_stats.npz'
    prepare_stats(args.split_root / 'train.csv', stats_path)
    train_set = LRCDataset(args.split_root / 'train.csv', stats_path, args.image_size)
    val_set = LRCDataset(args.split_root / 'val.csv', stats_path, args.image_size)
    test_set = LRCDataset(args.split_root / 'test.csv', stats_path, args.image_size)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)
    y = np.asarray([strict_label(r) for r in train_set.rows], dtype=np.int64)
    (pos, neg) = (int((y == 1).sum()), int((y == 0).sum()))
    if pos == 0 or neg == 0:
        raise RuntimeError(f'requires both classes, got P/N={pos}/{neg}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = ExplicitLRCModel(no_gs=args.no_gs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print('device:', device, 'no_gs:', args.no_gs)
    print('strict P/N:', pos, neg, 'pos_weight:', float(pos_weight))
    (best, best_path) = (-1.0, args.run_dir / 'best_explicit_lrc.pt')
    log_path = args.run_dir / 'train_log.csv'
    for epoch in range(1, args.epochs + 1):
        model.train()
        (total, count) = (0.0, 0)
        for batch in train_loader:
            b = move(batch, device)
            logit = model(b['rgb'], b['action_geom'], b['pointcloud'], b['left'], b['right'], b['corridor'], b['left_mask'], b['right_mask'], b['corridor_mask'])
            loss = criterion(logit, b['label'])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * len(logit)
            count += len(logit)
        sch.step()
        val = evaluate(model, val_loader, device, criterion)
        if val['mean_ap'] > best:
            best = float(val['mean_ap'])
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'best_score': best}, best_path)
        row = {'epoch': epoch, 'lr': opt.param_groups[0]['lr'], 'train_loss': total / max(count, 1), 'val_map': val['mean_ap'], 'val_top1': val['top1_success'], 'val_top5': val['top5_success'], 'val_auroc': val['auroc'], 'val_macro_f1': val['macro_f1'], 'best_val_map': best}
        with log_path.open('a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if epoch == 1:
                w.writeheader()
            w.writerow(row)
        print(f"[Epoch {epoch:03d}/{args.epochs}] loss={row['train_loss']:.4f} val_map={row['val_map']:.4f} top1={row['val_top1']:.4f} top5={row['val_top5']:.4f} auroc={row['val_auroc']:.4f} best={best:.4f}")
    state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(state['model'])
    test = evaluate(model, test_loader, device, criterion)
    (args.run_dir / 'best_val.json').write_text(json.dumps({'epoch': int(state['epoch']), 'selection_metric': 'val_mean_ap', 'best_score': float(state['best_score'])}, indent=2))
    (args.run_dir / 'test_metrics.json').write_text(json.dumps(test, indent=2))
    print('\n===== TEST =====')
    print(json.dumps(test, indent=2))
    print('DONE:', args.run_dir)


class PointCloudEncoder(torch.nn.Module):
    """
    PointNet-style fixed-view point-cloud encoder.

    Input:
        [B, 1024, 3] normalized VGGT XYZ points.

    Architecture:
        pointwise 3 -> 64 -> 128 -> 224
        symmetric max pooling
        224 -> 128

    The 224-channel layer makes the parameter count close to
    the removed GS endpoint/corridor/relation branch.
    """

    def __init__(self):
        super().__init__()

        self.point_mlp = torch.nn.Sequential(
            torch.nn.Conv1d(3, 64, 1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(64, 128, 1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(128, 224, 1),
            torch.nn.BatchNorm1d(224),
            torch.nn.ReLU(inplace=True),
        )

        self.projection = torch.nn.Sequential(
            torch.nn.Linear(224, 128),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, points):
        if points.ndim != 3:
            raise RuntimeError(
                f"POINTCLOUD_NDIM={points.ndim}"
            )

        if points.shape[-1] != 3:
            raise RuntimeError(
                f"POINTCLOUD_SHAPE={tuple(points.shape)}"
            )

        features = self.point_mlp(
            points.transpose(1, 2).contiguous()
        )

        pooled = torch.amax(features, dim=2)

        return self.projection(pooled)


class LRCDataset(_OriginalLRCDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._pointcloud_cache = {}

        rows = getattr(self, "rows", None)

        if not (
            isinstance(rows, list)
            and (
                len(rows) == 0
                or isinstance(rows[0], dict)
            )
        ):
            candidates = []

            for value in vars(self).values():
                if (
                    isinstance(value, list)
                    and len(value) == len(self)
                    and (
                        len(value) == 0
                        or isinstance(value[0], dict)
                    )
                ):
                    candidates.append(value)

            if len(candidates) != 1:
                raise RuntimeError(
                    "POINTCLOUD_DATASET_ROW_STORAGE_NOT_UNIQUE "
                    f"count={len(candidates)}"
                )

            rows = candidates[0]

        self._pointcloud_rows = rows

    def __getitem__(self, index):
        item = super().__getitem__(index)

        row = self._pointcloud_rows[int(index)]

        path = row.get(
            "vision_pointcloud_vggt_path",
            "",
        )

        if not path:
            raise RuntimeError(
                "MISSING_vision_pointcloud_vggt_path"
            )

        if path not in self._pointcloud_cache:
            import numpy as _np

            with _np.load(
                path,
                allow_pickle=False,
            ) as data:
                points = _np.asarray(
                    data["points"],
                    dtype=_np.float32,
                )

            if points.shape != (1024, 3):
                raise RuntimeError(
                    f"POINTCLOUD_ASSET_SHAPE="
                    f"{points.shape} path={path}"
                )

            if not _np.isfinite(points).all():
                raise RuntimeError(
                    f"POINTCLOUD_NONFINITE={path}"
                )

            self._pointcloud_cache[path] = (
                torch.from_numpy(points.copy())
            )

        item["pointcloud"] = (
            self._pointcloud_cache[path]
        )

        return item


class ExplicitLRCModel(torch.nn.Module):
    """
    RGB + Action + fixed-view VGGT point cloud.

    RGB:        256
    Action13:    64
    PointCloud: 128
    Fusion:     448
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        reference = _OriginalExplicitLRCModel(
            *args,
            **kwargs,
        )

        self.rgb_encoder = reference.rgb_encoder
        self.action_encoder = reference.action_encoder
        self.pointcloud_encoder = PointCloudEncoder()
        self.head = reference.head

    def forward(
        self,
        rgb,
        action_geom,
        pointcloud,
        left,
        right,
        corridor,
        left_mask,
        right_mask,
        corridor_mask,
    ):
        rgb_feature = self.rgb_encoder(rgb)

        action_feature = self.action_encoder(
            action_geom
        )

        pointcloud_feature = (
            self.pointcloud_encoder(pointcloud)
        )

        fused = torch.cat(
            [
                rgb_feature,
                action_feature,
                pointcloud_feature,
            ],
            dim=1,
        )

        return self.head(fused).squeeze(-1)


if __name__ == '__main__':
    main()
