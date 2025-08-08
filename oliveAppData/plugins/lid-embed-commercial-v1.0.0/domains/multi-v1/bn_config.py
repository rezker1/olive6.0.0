from idento3 import Pncc, FeatProcess, Feature

def wmvn_then_select_indices(data, indices=None):
    return Feature.select_indices(FeatProcess.mean_norm_win(data), indices).astype('f')

def nothing(data,indices=None):
    return data.astype('f')

def wmvn(data, indices=None):
    return FeatProcess.mean_norm_win(data).astype('f')

featclass   = Pncc
sample_rate = 8000 #16000
nfilters    = 40
ncep        = 20
window      = 200 #409
overlap     = 120 #249
fstart      = 200
fend        = 3400
addenergy   = False
includeC0   = True
preemphasis = None
min_rms     = 0.0
warpfact    = 0.0
bwscale     = 1.0
batch_sec   = 300.0
nfft        = 512
rootcomp    = 1./15.
ppn = False
spb = False

featprocess_func=wmvn
