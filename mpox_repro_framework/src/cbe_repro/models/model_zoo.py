# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Any, Dict

# tabular
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# add near other imports
import importlib

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# imaging head (classic features → linear classifier)
from sklearn.linear_model import LogisticRegression as ImgLogReg

@dataclass
class ModelSpec:
    name: str
    params: Dict[str, Any]

def make_tabular(spec: ModelSpec):
    n = spec.name.lower()
    p = spec.params or {}
    if n in ["logisticregression","logreg","lr"]:
        return LogisticRegression(**p)
    if n in ["randomforest","randomforestclassifier","rf"]:
        return RandomForestClassifier(**p)
    if n in ["svm","svc"]:
        return SVC(probability=True, **p)
    if n in ["xgboost","xgb","xgbclassifier"]:
        if not _HAS_XGB:
            raise ImportError("xgboost not installed. Try: pip install xgboost")
        return XGBClassifier(**p)
    raise ValueError(f"Unknown tabular model: {spec.name}")

def _make_vit_classifier(name: str, num_classes: int = 2, pretrained: bool = True, img_size: int = 224):
    """
    Returns (model, preprocess), where:
      - model: a timm ViT nn.Module
      - preprocess: torchvision.transforms pipeline matching ViT input size / stats
    """
    if importlib.util.find_spec("timm") is None:
        raise ImportError("timm not installed. pip install timm torchvision torch --upgrade")

    import timm
    import torch.nn as nn
    from torchvision import transforms

    # Map short names to timm names
    aliases = {
        "vit_b16": "vit_base_patch16_224",
        "vit-b/16": "vit_base_patch16_224",
        "vit_base_patch16_224": "vit_base_patch16_224",
    }
    timm_name = aliases.get(name.lower(), name)

    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    # (If model has no classifier set, timm handles num_classes; else replace head.)
    if hasattr(model, "head") and isinstance(model.head, nn.Module):
        # already set by create_model
        pass

    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    return model, preprocess


def make_imaging(model_spec: ModelSpec):
    """
    EXISTING function — add this branch at the top or before your defaults.
    """
    name = (model_spec.name or "").lower()
    p = model_spec.params or {}
    if name in ("vit_b16","vit-b/16","vit_base_patch16_224"):
        # return a tuple (model, preprocess, trainer_tag)
        # trainer_tag is a tiny hint so run_unified knows to use the torch trainer
        img_size = int(p.get("img_size", 224))
        model, preprocess = _make_vit_classifier(name, num_classes=2, pretrained=True,
                                                 img_size=int(model_spec.params.get("img_size", 224)))
        return {"engine": "torch_vit", "model": model, "preprocess": preprocess, "params": model_spec.params}
    
     # ---- Classic sklearn imaging heads ----
    if name in ("logisticregression", "logreg", "lr"):
        return ImgLogReg(**p)
    if name in ("randomforest", "randomforestclassifier", "rf"):
        return RandomForestClassifier(**p)
    if name in ("svm", "svc"):
        return SVC(probability=True, **p)
    if name in ("xgboost", "xgb", "xgbclassifier"):
        if not _HAS_XGB:
            raise ImportError("xgboost not installed. Try: pip install xgboost")
        return XGBClassifier(**p)

    raise ValueError(f"Unknown imaging model: {model_spec.name}")

    # ... keep your existing branches for logisticregression / RF / etc.
    # return sklearn-style models for feature-based imaging as you already do
