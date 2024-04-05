from segmentation.trainer import load_cfg, setup_cfg, Trainer
from detectron2.data.datasets import register_coco_instances
import os

INPUT_YOUR_VALUE = None


def register_dataset(args):
    register_coco_instances(
        "training", 
        {}, 
        INPUT_YOUR_VALUE, 
        INPUT_YOUR_VALUE
    )
    register_coco_instances(
        "validation", 
        {}, 
        INPUT_YOUR_VALUE, 
        INPUT_YOUR_VALUE
    )


def main(args):
    register_dataset(args)

    cfg = load_cfg(args.config, pretrained=args.pretrained)
    cfg = setup_cfg(cfg)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train detectron2")
    parser.add_argument("--base_dir", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--pretrained", action='store_true',help="load pretrained weights")
    parser.add_argument("--resume", action='store_true',help="resume training")

    args = parser.parse_args()

    from detectron2.engine import launch
    launch(
        main_func=main,
        num_gpus_per_machine= INPUT_YOUR_VALUE,
        num_machines= INPUT_YOUR_VALUE,
        machine_rank= 0,
        dist_url='auto',
        args= (args, )
        )
