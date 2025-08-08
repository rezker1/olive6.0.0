import os
curr_dir = os.path.dirname(os.path.realpath(__file__))
from idento3 import FeatProcess, Feature, TFDnnFeatures

featclass   = TFDnnFeatures
nnet_mvn = os.path.join(curr_dir, "bn_mvn.npz")
nnet = os.path.join(curr_dir, "bn_nnet.npz")
linearout   = True
layer       = 5
dnn_input_config = os.path.join(curr_dir, "bn_config.py")

def wmvn_then_select_indices(data, indices=None):
    return Feature.select_indices(FeatProcess.mean_norm_win(data), indices)

def nothing(data, indices=None):
    return data.astype('f')

featprocess_func=wmvn_then_select_indices
