from idento3 import Pncc, extend_array_borders, splice
import numpy as np

def wmvn(data, win):
    ndata = np.empty_like(data)
    nb_ex, dim = data.shape
    new_data = extend_array_borders(data, win, mode="reflect")
    for i in range(nb_ex):
        std      = np.std( new_data[i:i+win,:],axis=0)
        shift    = np.mean(new_data[i:i+win,:],axis=0)
        ndata[i,:] = (data[i,:] - shift)/std
    return ndata.astype('f')

def pp(feat, indices=None):
    window = 31
    window_norm = 201
    feat = feat[:,1:]
    
    feat = wmvn(feat.astype('f'), window_norm)

    if True:
        chunk_size = 1000*window
        compensation = (window-1)//2
        complete, chunk = 0, 10000
        while complete < feat.shape[0]:
            offset = compensation if complete>0 else 0
            cspliced = splice(feat[complete-offset:complete+chunk+compensation, :], window)[offset:chunk+offset].astype('f')
            yield cspliced
            complete += chunk


featclass   = Pncc
window      = 200
overlap     = 120
sample_rate = 8000
min_rms     = 0.0
min_duration = 0.31
normalize   = True
fstart      = 200
fend        = 3300
nfilters    = 40
ncep        = 21
nfft        = 512
preemphasis = None
addenergy   = False
includeC0   = True

featprocess_func = pp
