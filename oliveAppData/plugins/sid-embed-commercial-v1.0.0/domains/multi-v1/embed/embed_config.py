try:
    from idento import Pncc, FeatProcess, Feature
except:
    from idento3 import Pncc, FeatProcess, Feature

def wmvn_then_select_indices(data, indices=None):
    return Feature.select_indices(FeatProcess.mean_norm_win(data), indices).astype('f')

def nothing(data,indices=None):
    return data

featclass   = Pncc
sample_rate = 16000
nfilters    = 40
ncep        = 30
window      = 400
overlap     = 240
fstart      = 100
fend        = 7600
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
gain_norm_dynamic_limiter = True
gain_norm_dynamic_compression = False

featprocess_func=wmvn_then_select_indices
