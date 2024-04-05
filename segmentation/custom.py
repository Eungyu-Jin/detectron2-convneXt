import logging
import numpy as np
import os
import torch

from detectron2.data import transforms as T
from detectron2.data import build_detection_train_loader, MetadataCatalog
from detectron2.engine import HookBase
import detectron2.utils.comm as comm
from detectron2.data import detection_utils as utils

import operator
import copy


class CustomAugmentation(T.Augmentation):
    """
    ## CustomAugmentation
    albumentation compose wrapping해서 detectron2의 augmentation과 호환
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def get_transform(self, image):
        return CustomTransform(self.transforms)
    

class CustomTransform(T.Transform):
    def __init__(self, transforms):
        super().__init__()
        self._set_attributes(locals())
        self.transforms = transforms

    def apply_image(self, img):
        augmented_image = self.transforms(image=img)['image']
        return augmented_image

    def apply_coords(self, coords):
        #coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        #coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation)
        return segmentation

    def inverse(self):
        return CustomTransform(self.transforms)


class CustomMapper:
    """
    ## CustomMapper 
    Custom mapper used build_detection_train_loader. annotation을 detectron2 instance로 변환
    """
    def __init__(self, cfg, augmentations, image_format="BGR", recompute_boxes = False):
        self.cfg = cfg
        self.augmentations = augmentations  # image augentation compose 
        self.image_format = image_format
        self.recompute_boxes= recompute_boxes  # False if validation dataset

        logger = logging.getLogger(__name__)
        logger.info(f"[DatasetMapper] Augmentations used in {augmentations}")

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.augmentations, image)
        image_shape = image.shape[:2]  # h, w

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # COCO annotation 추출
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format="polygon"
        )
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict


class ValidationLoss(HookBase):
    """
    ## ValidationLoss
    Custom valdidation loss, using in Early Stopping tracking.
    """
    def __init__(self, cfg, mapper):
        super().__init__()
        self._cfg = copy.deepcopy(cfg)
        self._cfg.DATASETS.TRAIN = self._cfg.DATASETS.TEST
        self._loader = iter(build_detection_train_loader(self._cfg, mapper))
        
    def after_step(self):
        ## step 마다 val-loss 계산
        data = next(self._loader) 
        self.val_losses = []
        with torch.no_grad():
            loss_dict = self.trainer.model(data)
            
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict  # val_loss가 inf인 경우 assert

            loss_dict_reduced = {"val_" + k: v.item() for k, v in 
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            self.val_losses.append(losses_reduced)
            if comm.is_main_process():  # scalar 업데이트
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced, 
                                                 **loss_dict_reduced)


class CustomHook(HookBase):
    """
    customize hooks. Add val-loss monitor and early stoppings.

    ### args
    - monitor : monitor ONLY validation loss 
    - early_stopping
    - patience (epochs)
    """
    def __init__(
        self,
        checkpointer,
        monitor: str,
        early_stopping: bool,
        patience: int
    ) -> None:
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._checkpointer = checkpointer
        self._monitor = monitor
        self._patience = patience
        self._compare = operator.lt # val-loss는 minimize, val-acc는 maximize

        self.best_metric = None
        self.best_iter = None
        self._patience_counter = 0
        self._early_stopping = early_stopping  # cfg에서 early stopping enabled 여부

    def _update_best(self, val, iteration):
        # if math.isnan(val) or math.isinf(val):
        #     return False
        self.best_metric = val
        self.best_iter = iteration
        # return True

    def _checking(self):
        metric_tuple = self.trainer.storage.latest().get(self._monitor)  # scalar 읽어오기
        if metric_tuple is None:
            return
        else:
            latest_metric, metric_iter = metric_tuple

        if self.best_metric is None:
            self._update_best(latest_metric, metric_iter)
        elif self._compare(latest_metric, self.best_metric): # 이전 best 값과 비교
            additional_state = {"iteration": metric_iter}
            self._checkpointer.save("model_best", **additional_state)  # best 값 업데이트
            self._update_best(latest_metric, metric_iter)
            if self._early_stopping: 
                self._patience_counter = 0 
        else:
            if self._early_stopping:
                self._patience_counter += 1
                if self._patience_counter >= self._patience:  # early stopping 실행
                    additional_state = {"iteration": metric_iter}
                    self._checkpointer.save(f"model_last", **additional_state)
                    raise Exception
            else:
                pass

    def after_step(self):
        # step 이후 hooks
        self._checking()

    def after_train(self):
        # train 이후 hooks (학습 중단 시 last checkpointer로 돌아감)
        if self.trainer.iter + 1 >= self.trainer.max_iter:
            self._checking()
