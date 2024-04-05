import os
import itertools
import torch
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetMapper,
    build_detection_train_loader, 
    build_detection_test_loader, 
    MetadataCatalog, 
    DatasetCatalog
)
from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping
from detectron2.utils.env import TORCH_VERSION
import albumentations as A

from .add_config import add_custom_config
from .custom import CustomAugmentation, CustomHook, CustomMapper, ValidationLoss, build_evaluator


INPUT_YOUR_VALUE = None


def load_cfg(custom, pretrained=True):
    """
    Load custom cfg. If pretrained, use pretrained resneXt RCNN model.
    """
    cfg = get_cfg()
    
    if custom is not None:
        add_custom_config(cfg)
        cfg.merge_from_file(custom)

    if pretrained:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")

    return cfg


def round_step(x):
    """
    step -> epoch 계산할때 반올림
    """
    import math
    n_digit = int(math.log10(x)) + 1
    if n_digit < 3:
        return 0
    elif n_digit == 3:
        return round(x, -1)
    elif n_digit == 4:
        return round(x, -2)
    elif n_digit == 5:
        return round(x, -3)
    elif n_digit >= 6:
        return round(x, -4)


def setup_cfg(cfg):
    """
    데이터셋에 따른 cfg 자동 설정
    """
    cfg.SOLVER.IMS_LEN = sum([len(DatasetCatalog.get(d)) for d in cfg.DATASETS.TRAIN])
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(list(itertools.chain(*[MetadataCatalog.get(d).thing_classes for d in cfg.DATASETS.TRAIN]))) 

    num_gpus = torch.cuda.device_count()
    if num_gpus ==0:
        cfg.MODEL.DEVICE='cpu'
    else:
        cfg.SOLVER.NUM_GPUS= num_gpus
    
    iter_per_epoch = cfg.SOLVER.IMS_LEN // cfg.SOLVER.IMS_PER_BATCH # 1 epochs 당 iterations 수
    max_iter = iter_per_epoch * cfg.SOLVER.EPOCHS # 최대 iterations 수
    cfg.SOLVER.MAX_ITER = max_iter

    if cfg.SOLVER.STEPS != ():
        cfg.SOLVER.STEPS = (round_step(int(max_iter*0.75)), round_step(int(max_iter*0.9))) # LR steps

    # epochs 단위로 바꿈
    cfg.SOLVER.EARLY_STOPPING.PATIENCE = cfg.SOLVER.EARLY_STOPPING.PATIENCE * iter_per_epoch
    cfg.SOLVER.CHECKPOINT_PERIOD = cfg.SOLVER.CHECKPOINT_PERIOD * iter_per_epoch
    cfg.TEST.EVAL_PERIOD = cfg.TEST.EVAL_PERIOD * iter_per_epoch

    return cfg


def compose_augementation(cfg, is_train):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"

    resize = [
        T.ResizeShortestEdge(min_size, max_size, sample_style)
    ]

    if is_train:
        ## pixel 증강은 기능이 많은 albumentation 사용
        pixel_compose = [
            A.OneOf(
                [
                    A.RandomBrightnessContrast(INPUT_YOUR_VALUE),  # 밝기대비
                    A.GaussNoise(INPUT_YOUR_VALUE), 
                ],
                p=INPUT_YOUR_VALUE
            )
        ]
        # detectron2에 적용되도록 CustomAugmentation wrapping
        pixel = [
            CustomAugmentation(
                transforms=A.Compose(pixel_compose)
            )
        ]

        # affine 변환은 detectron2의 transform 사용
        affine = [
            T.RandomFlip(INPUT_YOUR_VALUE),
            T.RandomApply(
                T.RandomCrop(
                    crop_type=INPUT_YOUR_VALUE,
                    crop_size=INPUT_YOUR_VALUE,
                ),
                prob=INPUT_YOUR_VALUE
            )
        ]
        
        augs = pixel + affine + resize
    else:
        augs = resize

    return augs


class Trainer(DefaultTrainer):
    """
    ## Trainer
    Build Custom trainer and hooks.
    """
    @classmethod
    def build_train_loader(cls, cfg):
        augs = compose_augementation(cfg=cfg, is_train=True)
        mapper = CustomMapper(cfg=cfg, augmentations=augs, recompute_boxes=True)

        return build_detection_train_loader(cfg, mapper=mapper)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        augs = compose_augementation(cfg=cfg, is_train=False)
        mapper = CustomMapper(cfg=cfg, augmentations=augs, recompute_boxes=False)

        return build_detection_test_loader(cfg, dataset_name, mapper = mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)
    
    @classmethod
    def build_optimizer(cls, cfg, model):
        ## AdanW 추가
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        )

        optim_args = {}

        if TORCH_VERSION >= (1, 12):
            optim_args["foreach"] = True

        if cfg.SOLVER.OPTIM == 'AdamW':
            optim_args.update({
                "params": params,
                "lr": cfg.SOLVER.BASE_LR,
                "amsgrad": cfg.SOLVER.AMSGRAD,
                "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
            })
            optim = torch.optim.AdamW(**optim_args)
        else:
            optim_args = ({
                "params": params,
                "lr": cfg.SOLVER.BASE_LR,
                "momentum": cfg.SOLVER.MOMENTUM,
                "nesterov": cfg.SOLVER.NESTEROV,
                "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
            })

            optim = torch.optim.SGD(**optim_args)

        return maybe_add_gradient_clipping(cfg, optim)

    def build_hooks(self):
        ## training hooks 
        ret = super().build_hooks()

        augs = compose_augementation(cfg=self.cfg, is_train=False)
        mapper = CustomMapper(cfg=self.cfg, augmentations=augs, recompute_boxes=False)

        custom = [
            CustomHook(
                checkpointer=self.checkpointer,
                monitor= self.cfg.SOLVER.MONITOR,
                early_stopping=self.cfg.SOLVER.EARLY_STOPPING.ENABLED,
                patience=self.cfg.SOLVER.EARLY_STOPPING.PATIENCE
            ),
            ValidationLoss(
                cfg = self.cfg,
                mapper= mapper
            )
        ]
        ret.extend(custom)

        return ret
    
    
def align_hooks(trainer):
    """align custom hooks"""
    trainer._hooks = trainer._hooks[:4] + trainer._hooks[4:][::-1]
