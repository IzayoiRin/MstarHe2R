import os

"""Base Configures"""
# Base Core Application Root Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Source Data Root Path
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
# Format for Img to Tensor
IMG_TENSOR_FORMAT = "chanel_first"
# Img size H * W
IMG_SIZE = 128 * 128
# Device for GPU CUDA
CUDA_DEVICE_AVAILABLE = True


"""Script MstarTransFormatter Parameters"""
# raw STAR data dir
RAW_DATA_DIR = "Rdata"
# transformed STAR data dir
TRANS_OUT_DIR = "Odata"
# trans formats
TRANS2FORMATS = ("jpg", "tensor")
# trans parameters
TRANS2PARAMS = {
    "float_size": 4,
    "img_format": "chanel_last",
    "gamma": 2.2
}


"""Script MstarDataInitiation Parameters"""
# tensor MSTAR  data dir
TENSOR_DATA_DIR = "TensorData"
# standard tensor MSTAR DataSet cache dir
MSTAR_SAVE_DIR = "STDataSet"
# Using MSTAR top dir
MSTAR_PREFIXED_DIR = "adjust3"
# MSTAR data model
MSTAR_DATA_MODE_DIR = ("TEST", "TRAIN")
# tensor MSTAR file for different model, One search pattern as single Class
MSTAR_DATA_TARGETS = {
        1: [
            "17_DEG@BMP2@\D+9563@.*",
            "17_DEG@BTR70@.*@.*",
            "17_DEG@T72@\D+132@.*",
        ],
        0: [
            "15_DEG@BMP2@\D+9563@.*",
            "15_DEG@BTR70@.*@.*",
            "15_DEG@T72@\D+132@.*",
        ]
    }


"""MstarDataLoader Parameters"""
# Training batch size
MSTAR_BATCH_SIZE = 64
# Shuffle batch
LOADER_SHUFFLE = True
# Validating data size
VALIDATE_RATE = 0.2
# Reload MSTAR DataSet
FLUSH_MSTAR = False


"""Models Parameters"""
COMPUTATION_GRAPHS = "components.graphs"
PERSISTENCE_DAT_DIR = os.path.join(os.path.dirname(BASE_DIR), "persistence_models")
MODEL_LOADER_PARAMS = {
    "train": {
        "batch_size": MSTAR_BATCH_SIZE,
    },
    "test": {
        "batch_size": MSTAR_BATCH_SIZE,
    }
}
# Hyper parameters of training in different graphs
# dict[-aph: Early stop threshold, -n: Max epoch, -step: Max MP recording step, -cp: Checkpoint frequency]
HYPER_PARAMETERS = {
    "default": dict(aph=0.5, n=200, stp=10, cp=2),
    "graph0": {
        "MSTARCNNetGraph": dict(aph=0.5, n=200, stp=10, cp=2),
    },
    "graph1": {
        "L4MSTARANNetGraph": dict(aph=0.5, n=500, stp=10, cp=2),
        # "L4MSTARANNetGraphV2": dict(aph=0.5, n=600, stp=10, cp=2),
        # "L2MSTARANNetGraph": dict(aph=0.5, n=700, stp=10, cp=2),
        # "L5MSTARANNetGraph": dict(aph=0.5, n=200, stp=10, cp=2),
    },
    "graph2": {
        "TestL4MSTARANNetGraph": dict(aph=0.5, n=100, stp=10, cp=0),
    },
}


"""Entry Function Configures"""
# Command from shell
# [dict[-k: cmd arg, -v: entry function [-default: main]]]
COMMANDS = {
    # only for test case
    "test": "apps.test",
    # trans original mstar SRA file to jpg or tensor file
    "transmstar": "apps.mstarFormatter",
    # initial MSTAR DATASET object cache
    "initmstar": "apps.mstarDataInitiation",
    # MSTAR training -d: device [c, g]
    "train": "apps.mstarTrain",
}
