# this script runs image segmentation of AMF dataset based by transfer learning
# Mask-RCNN on detectron2:
import argparse
import os
import random
import shutil
import time
import warnings
import pickle
import numpy as np
import math
import sys
import copy
import re
import pandas as pd
import matplotlib.pyplot as plt
import json
import cv2

import torch
import torch.nn as nn
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor,DefaultTrainer,HookBase
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer,ColorMode
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator,inference_on_dataset
from detectron2.data import build_detection_test_loader,DatasetMapper,build_detection_train_loader,MetadataCatalog,DatasetCatalog
import detectron2.data.transforms as T
import detectron2.utils.comm as comm
# default Arguments
args_internal_dict={
    "batch_size": (2,int),
    "epochs": (300,int),
    "learning_rate": (0.00025,float),
    # "no_cuda": (False,bool),
    "seed": (1,int),
    "net_struct": ("mask_rcnn_R_50_FPN_3x",str),
    # "optimizer": ("adam",str),##adam
    "gpu_use": (1,int),# whehter use gpu 1 use 0 not use
    "freeze_at": (2,int), #till n block ResNet18 has 10 blocks
    "aug_flag": (1,int)#whether do the more comprehensive augmentation (1) or not (0)
}
def build_aug(cfg):
    augs=[T.ResizeShortestEdge(short_edge_length=(640,672,704,736,768,800),max_size=1333,sample_style='choice'),T.RandomBrightness(0.5,2.0),T.RandomCrop("relative_range",[0.5,0.5]),T.RandomFlip(),T.RandomRotation([0,360])]
    return augs

class ValidationLoss_checkpoint(HookBase):
    # adapted from https://github.com/facebookresearch/detectron2/issues/810
    # https://github.com/facebookresearch/detectron2/issues/2114
    # validatiion is done for each batch and summarized for a few iteration (smaller than epoch, similar to train)
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg.clone()# a local config for validaiton
        self.cfg.DATASETS.TRAIN=self.cfg.DATASETS.VAL
        # self.cfg.SOLVER.IMS_PER_BATCH=1
        # self.cfg.SOLVER.IMS_PER_BATCH=self.cfg.VALSIZE
        self._loader=iter(build_detection_train_loader(self.cfg))
    
    def after_step(self):
        data=next(self._loader)
        with torch.no_grad():
            loss_dict=self.trainer.model(data)
            losses=sum(loss_dict.values())
            assert torch.isfinite(losses).all(),loss_dict
            loss_dict_reduced={"val_" + k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced=sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced,**loss_dict_reduced)
            # the best loss by validation set
            metric=self.trainer.storage.history('total_val_loss')
            stor_val=metric.values()
            vec=[ele[0] for ele in stor_val]
            newind=stor_val[len(stor_val)-1][1]
            is_best=vec.index(min(vec))==newind
            if is_best:
                self.trainer.checkpointer.save("model_best")

def parse_func_wrap(parser,termname,args_internal_dict):
    commandstring='--'+termname.replace("_","-")
    defaulval=args_internal_dict[termname][0]
    typedef=args_internal_dict[termname][1]
    parser.add_argument(commandstring,type=typedef,default=defaulval,
                        help='input '+str(termname)+' for training (default: '+str(defaulval)+')')
    
    return(parser)

def get_amseg_dicts(img_dir,classes):
    anno_file=os.path.join(img_dir,"regiondata.csv")
    annotab=pd.read_csv(anno_file,delimiter="\t")
    files=annotab['filename'].unique()
    dataset_dicts=[]
    for idx,file in enumerate(files):
        record={}
        filename=os.path.join(img_dir,file)
        height,width=cv2.imread(filename).shape[:2]
        record["file_name"]=filename
        record["image_id"]=idx
        record["height"]=height
        record["width"]=width
        subtab=annotab[annotab['filename']==file]
        objs=[]
        for anno_i in range(subtab.shape[0]):#multiple masks/boxes
            tab_rec=subtab.iloc[anno_i]
            # assert not tab_rec["region_attributes"]#check it is []
            anno=json.loads(tab_rec["region_shape_attributes"])
            if len(anno)==0:
                continue
            
            px=anno["all_points_x"]
            py=anno["all_points_y"]
            poly=[(x+0.5,y+0.5) for x,y in zip(px,py)]
            poly=[p for x in poly for p in x]
            category_id=np.where([ele==tab_rec['region_attributes'] for ele in classes])[0][0]
            obj={
                "bbox":[np.min(px),np.min(py),np.max(px),np.max(py)],
                "bbox_mode":BoxMode.XYXY_ABS,
                "segmentation":[poly],
                "category_id":category_id,
            }
            objs.append(obj)
        
        record["annotations"]=objs
        dataset_dicts.append(record)
    
    return dataset_dicts

class newtrainer(DefaultTrainer):
    @classmethod
    # def build_evaluator(cls,cfg,dataset_name,output_folder=None):
    #     if output_folder is None:
    #         output_folder=os.path.join(cfg.OUTPUT_DIR,"validation")
    #     return COCOEvaluator(dataset_name,("bbox","segm"),True,output_folder)
    def build_train_loader(cls,cfg):
        if cfg.AUG_FLAG==1:
            mapper=DatasetMapper(cfg,is_train=True,augmentations=build_aug(cfg))
        else:
            mapper=DatasetMapper(cfg,is_train=True)
        
        return build_detection_train_loader(cfg,mapper=mapper)

# passing arguments
parser=argparse.ArgumentParser(description='PyTorch Example')
for key in args_internal_dict.keys():
    parser=parse_func_wrap(parser,key,args_internal_dict)

args=parser.parse_args()
classes=['root','AMF internal hypha','AMF external hypha','AMF arbuscule','AMF vesicle','AMF spore','others']
for direc in ["train","validate",'test']:
    DatasetCatalog.register("am_"+direc,lambda direc=direc: get_amseg_dicts("../data/AM_classify2/"+direc,classes))
    MetadataCatalog.get("am_"+direc).set(thing_classes=classes)#classes name list

# configuration parameters
cfg=get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/"+args.net_struct+".yaml"))
cfg.DATASETS.TRAIN=("am_train",)
cfg.DATASETS.VAL=("am_validate",)#("",)
# cfg.VALSIZE=len(get_amseg_dicts("../data/AM_classify2/validate"))
trainsize=len(get_amseg_dicts("../data/AM_classify2/train",classes))
cfg.DATASETS.TEST=()
# cfg.TEST.EVAL_PERIOD=20
cfg.DATALOADER.NUM_WORKERS=2
cfg.MODEL.WEIGHTS=model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/"+args.net_struct+".yaml")#Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH=args.batch_size
cfg.SOLVER.BASE_LR=args.learning_rate
cfg.SOLVER.MAX_ITER=np.int(args.epochs*trainsize/args.batch_size) #iterations or epochs
if args.gpu_use!=1:
    cfg.MODEL.DEVICE='cpu'

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE=128#Number of regions per image used to train RPN. faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES=len(classes)# (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.BACKBONE.FREEZE_AT=args.freeze_at
# cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS=True
cfg.SEED=args.seed
cfg.AUG_FLAG=args.aug_flag
#
os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)
trainer=newtrainer(cfg)
val_loss_checkp=ValidationLoss_checkpoint(cfg)
trainer.register_hooks([val_loss_checkp])
# swap the order of PeriodicWriter and ValidationLoss
trainer._hooks=trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
trainer.resume_or_load(resume=False)
trainer.train()#training
# inference
cfg.MODEL.WEIGHTS=os.path.join(cfg.OUTPUT_DIR,"model_best.pth")# path to the model we just trained
trainer_val=newtrainer(cfg)
trainer_val.resume_or_load(resume=False)
# performance on training set
evaluator_train=COCOEvaluator("am_train",("bbox","segm"),False,output_dir="./output/")
train_loader=build_detection_test_loader(cfg,"am_train")
print(inference_on_dataset(trainer_val.model,train_loader,evaluator_train))
#
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.7# set a custom testing threshold
predictor=DefaultPredictor(cfg)

# validation data set
am_metadata_val=MetadataCatalog.get("am_validate")
dataset_dicts=get_amseg_dicts("../data/AM_classify2/validate",classes)
# random viszualize of 10 images
imageset=random.sample(dataset_dicts,10)
for d in imageset:
    im=cv2.imread(d["file_name"])
    outputs=predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    # prediction
    v=Visualizer(im[:,:,::-1],
                   metadata=am_metadata_val,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out=v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite('showimage.exp.pred'+str(d['image_id'])+'_validate.jpg',out.get_image()[:, :, ::-1])
    # ground truth
    v=Visualizer(im[:,:,::-1],
                   metadata=am_metadata_val,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW
    )
    out=v.draw_dataset_dict(d)
    cv2.imwrite('showimage.exp.groundtruth'+str(d['image_id'])+'_validate.jpg',out.get_image()[:, :, ::-1])
#
evaluator_val=COCOEvaluator("am_validate",("bbox","segm"),False,output_dir="./output/")
val_loader=build_detection_test_loader(cfg,"am_validate")
print(inference_on_dataset(trainer_val.model,val_loader,evaluator_val))

# test data set
am_metadata_test=MetadataCatalog.get("am_test")
dataset_dicts=get_amseg_dicts("../data/AM_classify2/test",classes)
# random viszualize of 10 images
imageset=random.sample(dataset_dicts,10)
for d in imageset:
    im=cv2.imread(d["file_name"])
    outputs=predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    # prediction
    v=Visualizer(im[:,:,::-1],
                   metadata=am_metadata_test,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out=v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite('showimage.exp.pred'+str(d['image_id'])+'_test.jpg',out.get_image()[:, :, ::-1])
    # ground truth
    v=Visualizer(im[:,:,::-1],
                   metadata=am_metadata_test,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW
    )
    out=v.draw_dataset_dict(d)
    cv2.imwrite('showimage.exp.groundtruth'+str(d['image_id'])+'_test.jpg',out.get_image()[:, :, ::-1])
#
evaluator_test=COCOEvaluator("am_test",("bbox","segm"),False,output_dir="./output/")
test_loader=build_detection_test_loader(cfg,"am_test")
print(inference_on_dataset(trainer_val.model,test_loader,evaluator_test))

# vis: tensorboard --logdir ./dir
