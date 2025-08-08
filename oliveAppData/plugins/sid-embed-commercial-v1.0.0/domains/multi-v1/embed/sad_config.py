import idento3 as idt
import numpy as np

featclass   = idt.KaldiMfcc
sample_rate = 8000
nfilters    = 40
fstart      = 200
fend        = 3300
gain_norm   = True
preemphasis = 0.97
ncep        = 21
includeC0   = False
 
def wmvn(data, win):
    ndata = np.empty_like(data)
    nb_ex, dim = data.shape
    new_data = idt.extend_array_borders(data, win, mode="reflect")
    for i in range(nb_ex):
        std      = np.std( new_data[i:i+win,:],axis=0)
        shift    = np.mean(new_data[i:i+win,:],axis=0)
        ndata[i,:] = (data[i,:] - shift)/std
    return ndata.astype('f')

def vad_pp(feat, indices=None):
    window = 31
    window_norm = 201
    feat = feat[:,1:]

    feat = wmvn(feat, window_norm).astype('f')
    compensation = int((window-1)/2)
    complete, chunk = 0, 10000
    while complete < feat.shape[0]:
        offset = compensation if complete>0 else 0
        cspliced = idt.splice(feat[complete-offset:complete+chunk+compensation, :], window)[offset:chunk+offset].astype('f')
        yield cspliced
        complete += chunk

featprocess_func = vad_pp
