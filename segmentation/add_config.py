from detectron2.config import CfgNode as CN

def add_convnext_config(cfg):
    _C = cfg
    _C.MODEL.CONVNEXT=CN()
    _C.MODEL.CONVNEXT.DEPTHS = [3, 3, 27, 3]
    _C.MODEL.CONVNEXT.DIMS = [256, 512, 1024, 2048]
    _C.MODEL.CONVNEXT.DROP_PATH_RATE = 0.8
    _C.MODEL.CONVNEXT.LAYER_SCALE_INIT_VALUE = 1.0
    _C.MODEL.CONVNEXT.OUT_FEATURES = [0,1,2,3]

def add_solver_config(cfg):
    _C = cfg
    _C.SOLVER.EPOCHS = 20
    _C.SOLVER.IMS_LEN = 10000
    _C.SOLVER.OPTIM = 'SGD'
    _C.SOLVER.AMSGRAD = False
    _C.SOLVER.EARLY_STOPPING = CN()
    _C.SOLVER.EARLY_STOPPING.ENABLED = False
    _C.SOLVER.EARLY_STOPPING.PATIENCE = 0
    _C.SOLVER.MONITOR = 'total_val_loss'
    
def add_custom_config(cfg):
    add_convnext_config(cfg)
    add_solver_config(cfg)