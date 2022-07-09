"""
Main program entry point.

The script contains training and testing codes based on Detectron2.

Modified from https://github.com/facebookresearch/detectron2/blob/main/tools/train_net.py

"""
__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = ["Detectron2"]
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"

import os, sys

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import verify_results
from detectron2.checkpoint import DetectionCheckpointer

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from d2sgg.config import add_sgg_config
from d2sgg.data.datasets import register_visual_genome
from d2sgg.engine import SGTrainer
from d2sgg.misc import show_latency

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_sgg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)

    register_visual_genome(cfg)

    if args.eval_only:
        model = SGTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = SGTrainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.MODEL.RELATION_ON:
            show_latency(model)
        return res

    trainer = SGTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
