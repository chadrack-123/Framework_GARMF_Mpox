# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional  # <-- add this

class ImageFolderDataset:
    """
    Minimal, dependency-light loader for mpox lesion images.
    - Expects ImageFolder-style structure: root/{split}/{class}/*.jpg
    - Deterministic augmentations for reproducible "GenAI-style" oversampling
    - Simple featurization: downsample + color histograms (works with sklearn)
    """
    def __init__(self, root: str, split: str,
                 pos_cls: str = "mpox", neg_cls: str = "non_mpox",
                 seed: int = 1337, img_size: int = 128):
        self.root = Path(root)
        self.split = split
        self.pos_cls = pos_cls
        self.neg_cls = neg_cls
        self.img_size = img_size
        self.rng = np.random.default_rng(seed)
        self.samples, self.labels = self._collect()

    def _collect(self):
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        paths, labels = [], []
        for label, cls in [(1, self.pos_cls), (0, self.neg_cls)]:
            for p in (self.root / self.split / cls).glob("*"):
                if p.suffix.lower() in exts:
                    paths.append(p)
                    labels.append(label)
        return np.array(paths), np.array(labels, dtype=int)

    def __len__(self):
        return len(self.samples)

    # ---------- I/O + deterministic aug ----------

    def _load_image(self, path: Path) -> Image.Image:
        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_size, self.img_size))
        return img

    def _deterministic_augment(self, img: Image.Image, idx: int, enable: bool = False) -> Image.Image:
        """A tiny, deterministic augmentor controlled by sample index."""
        if not enable:
            return img
        m = idx % 4
        if m == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif m == 2:
            img = img.rotate(90, expand=False)
        elif m == 3:
            img = img.rotate(270, expand=False)
        return img

    # ---------- simple features ----------

    def _simple_features(self, img: Image.Image) -> np.ndarray:
        """
        Features = downsampled pixels + 3x16-bin color histograms.
        Keeps it light so we can use scikit-learn models deterministically.
        """
        arr = np.asarray(img, dtype=np.float32) / 255.0  # [H,W,3]
        # 3 histograms (R/G/B) with 16 bins each
        hist_r, _ = np.histogram(arr[:, :, 0], bins=16, range=(0, 1), density=True)
        hist_g, _ = np.histogram(arr[:, :, 1], bins=16, range=(0, 1), density=True)
        hist_b, _ = np.histogram(arr[:, :, 2], bins=16, range=(0, 1), density=True)
        # downsample (32x32x3 if img_size=128) and flatten
        flat = arr[::4, ::4, :].reshape(-1)
        feats = np.concatenate([flat, hist_r, hist_g, hist_b]).astype(np.float32)
        return feats

    # ---------- public API ----------
    def as_features_labels(              # <-- now indented inside the class
        self,
        synth_enabled: bool = False,
        synth_multiplier: float = 2.0,     # legacy: oversample positives only
        balance_to_max: bool = False,      # fully balance minority up to majority
        target_ratio: Optional[float] = None,  # minority ≈ target_ratio * majority
        seed: int = 1337,
        synth_verbose: bool = True,
    ):
        """
        Returns (X, y) as numpy arrays.
        Modes (checked in order):
          1) If synth_enabled and (balance_to_max or target_ratio is not None):
               oversample the true minority class to the desired count using
               deterministic augmentations (for BOTH classes, not just positives).
          2) Else if synth_enabled and synth_multiplier > 1.0:
               legacy behavior — oversample positives (label==1) only.
          3) Else: no oversampling.
        """
        rng = np.random.default_rng(seed)

        # ----- base features (no aug) -----
        X, y = [], []
        for i, p in enumerate(self.samples):
            img = self._load_image(p)
            img = self._deterministic_augment(img, i, enable=False)
            X.append(self._simple_features(img))
            y.append(int(self.labels[i]))

        if len(X) == 0:
            raise FileNotFoundError(
                "No images found for features in this split. "
                f"root={self.root} split={self.split} "
                f"pos='{self.pos_cls}' neg='{self.neg_cls}'. "
                "Check folder names and that images exist with supported extensions."
            )
        X = np.vstack(X).astype(np.float32)
        y = np.array(y, dtype=int)

        if not synth_enabled:
            return X, y

        # helper: add N more samples from a given label using deterministic aug
        def _augment_label(label: int, need: int):
            if need <= 0:
                return []
            idxs = np.where(y == label)[0]
            if len(idxs) == 0:
                return []
            rep = (need + len(idxs) - 1) // len(idxs)
            take = np.tile(idxs, rep)[:need]
            aug_feats = []
            for k, idx in enumerate(take):
                img = self._load_image(self.samples[idx])
                img = self._deterministic_augment(img, k, enable=True)
                aug_feats.append(self._simple_features(img))
            return aug_feats

        counts = dict(zip(*np.unique(y, return_counts=True)))
        if synth_verbose:
            print(f"[IMG] counts before synth: {counts}")

        # Mode 1: balancing knobs
        if balance_to_max or (target_ratio is not None):
            maj_label = max(counts, key=counts.get)
            maj_n = counts[maj_label]
            ratio = 1.0 if balance_to_max else float(target_ratio)

            desired = {}
            for lbl, n in counts.items():
                if lbl == maj_label and ratio >= 1.0:
                    desired[lbl] = maj_n
                else:
                    desired[lbl] = max(n, int(np.ceil(ratio * maj_n)))

            X_add, y_add = [], []
            for lbl, want in desired.items():
                need = want - counts[lbl]
                if need > 0:
                    feats = _augment_label(lbl, need)
                    if feats:
                        X_add.append(np.vstack(feats).astype(np.float32))
                        y_add.append(np.full(len(feats), lbl, dtype=int))
            if X_add:
                X = np.vstack([X] + X_add)
                y = np.concatenate([y] + y_add)

            if synth_verbose:
                counts_after = dict(zip(*np.unique(y, return_counts=True)))
                print(f"[IMG] balancing → ratio={ratio} before={counts} after={counts_after}")
            return X, y

        # Mode 2: legacy positive-only oversample
        if synth_multiplier > 1.0:
            pos_idx = np.where(y == 1)[0]
            if len(pos_idx) > 0:
                target = int(len(pos_idx) * synth_multiplier)
                extra = target - len(pos_idx)
                feats = _augment_label(1, extra)
                if feats:
                    X = np.vstack([X, np.vstack(feats).astype(np.float32)])
                    y = np.concatenate([y, np.ones(len(feats), dtype=int)])
            if synth_verbose:
                counts_after = dict(zip(*np.unique(y, return_counts=True)))
                print(f"[IMG] legacy positive oversample → mult={synth_multiplier} "
                      f"before={counts} after={counts_after}")
        return X, y
