import os
from .dataset_h5 import H5PanDataset, H5PanDataset_test

def get_data(cfg, data_name):
    base_path = cfg['data']['base_path']
    data_name = cfg['data']['data_name']
    if "wv3" in data_name:
        data_train = "train_wv3.h5"
    elif "gf2" in data_name:
        data_train = "train_gf2.h5"
    elif "qb" in data_name:
        data_train = "train_qb.h5"
    file_path = os.path.join(base_path, data_train)
    return H5PanDataset(file_path)

def get_val_data(cfg, data_name):
    base_path = cfg['data']['base_path']
    data_name = cfg['data']['data_name']
    if "wv3" in data_name:
        data_val = "valid_wv3.h5"
    elif "gf2" in data_name:
        data_val = "valid_gf2.h5"
    elif "qb" in data_name:
        data_val = "valid_qb.h5"
    val_file_path = os.path.join(base_path, data_val)
    return H5PanDataset_test(val_file_path)

def get_test_data(cfg, data_name):
    base_path = cfg['data']['base_path']
    data_name = cfg['data']['data_name']
    if "wv3" in data_name:
        data_test = "test_wv3_multiExm1.h5"
    elif "gf2" in data_name:
        data_test = "test_gf2_multiExm1.h5"
    elif "qb" in data_name:
        data_test = "test_qb_multiExm1.h5"
    test_file_path = os.path.join(base_path, data_test)
    return H5PanDataset_test(test_file_path)

