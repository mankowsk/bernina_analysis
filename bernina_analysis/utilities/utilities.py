import numpy as np
import dask.array as da
import sys
sys.path.insert(0,'/sf/bernina/config/src/python/escape_dev/')
import escape.parse.swissfel as sf
import h5py
import escape as esc
import scipy

### DATA CORRECTION AND MANIPULATION FUNCTIONS FOR ESCAPE ARRAYS ###
def mod_pid(ea,n):
    index = ea.index +n
    data = ea.data
    step_lengths = ea.scan.step_lengths
    parameter = ea.scan.parameter
    return esc.Array(data=data, index=index, step_lengths=step_lengths, parameter=parameter)

def correct_ioxos_shape(ea):
    return esc.Array(data=ea.data.T[0], index = ea.index, parameter = ea.scan.parameter, step_lengths = ea.scan.step_lengths) 

def create_ea_from_data(ea, data):
    return esc.Array(data=data, index = ea.index, parameter = ea.scan.parameter, step_lengths = ea.scan.step_lengths) 

def calc_i0_pos_ea(ch4, ch5, ch6, ch7):
    data_i0_x = 4.133*(ch4.data*0.001609-0.00139*ch5.data)/(ch4.data*0.00161+0.00139*ch5.data)
    data_i0_y =-3.997*(ch6.data*0.00208-0.00232*ch7.data)/(ch6.data*0.00208+0.00232*ch7.data)
    i0_x = esc.Array(data=data_i0_x, index = ch4.index, parameter = ch4.scan.parameter, step_lengths = ch4.scan.step_lengths) 
    i0_y = esc.Array(data=data_i0_y, index = ch4.index, parameter = ch4.scan.parameter, step_lengths = ch4.scan.step_lengths) 
    return {'i0_x': i0_x, 'i0_y': i0_y}

def on_off(data,evt, compute=False,invert=False):
    laser_delayed = evt[:,25]==1
    if invert:
        laser_delayed = evt[:,25]==0
    for ei, d in enumerate(data):
        data[ei] = dict(off = data[ei][laser_delayed] ,on = data[ei][~laser_delayed])
        if compute:
            if ei == 0: 
                data[ei].update({'Id_on': data[ei]['on'].index, 'Id_off': data[ei]['off'].index})
            for las in ['on','off']:
                try:
                    data[ei][las] =  data[ei][las].data.compute()
                except AttributeError:
                    data[ei][las] =  data[ei][las].data             
    return data

##########################################################################


### TIMETOOL ANALYSIS SCRIPTS ###

def analyse_edge_correlation(tt_sig, ratio_av=None, roi=[650,1350]):
    #"""tt_sig: dictionary with keys 'on' and 'off', which hold 1d escape arrays of timetool traces with and without overlapped laser and x-ray pulses. The script correlates an average of 100 edges with the individual shots and returns two escape arrays, holding the position of the edge in px and the amplitude of the correlation peak."""
    tt_sig['off_sm'] = tt_sig['off'].map_index_blocks(scipy.ndimage.uniform_filter, size=(10,10))
    tt_sig['on_sm'] = tt_sig['on'].map_index_blocks(scipy.ndimage.uniform_filter, size=(1,10))
    idx = np.digitize(tt_sig['on'].index, tt_sig['off'].index[:-1]-0.5)
    tt_ratio_sm = esc.Array(index=tt_sig['on_sm'].index, data=tt_sig['on_sm'].data/tt_sig['off_sm'].data[idx]-1, step_lengths=tt_sig['on_sm'].scan.step_lengths, parameter =tt_sig['on_sm'].scan.parameter )
    corr = tt_ratio_sm.map_index_blocks(scipy.ndimage.correlate1d, ratio_av[roi[0]:roi[1]], axis=1, dtype=np.float64)
    corr = corr.compute()
    corr_amp = np.max(corr.data, axis=1)
    corr_pos = np.argmax(corr.data, axis=1)
    corr_pos_ea = create_ea_from_data(ea=tt_sig['on'], data=corr_pos)
    corr_amp_ea = create_ea_from_data(ea=tt_sig['on'], data=corr_amp)
    return corr_pos_ea, corr_amp_ea

def find_edge(data, step_length=50, edge_type='falling', step_type='heaviside', refinement=1, scale=10):
    # refine datacurrent_ref_correction 
    if data.ndim == 1:
        data = data[np.newaxis, :]
    def _interp(fp, xp, x):  # utility function to be used with apply_along_axis
        return np.interp(x, xp, fp)
    data_length = data.shape[1]
    refined_data = np.apply_along_axis(_interp, axis=1, arr=data, x=np.arange(0, data_length - 1, refinement),xp=np.arange(data_length),)
                                                
    # prepare a step function and refine it
    if step_type == 'heaviside':
        step_waveform = np.ones(shape=(step_length,))
        if edge_type == 'rising':
            step_waveform[: int(step_length / 2)] = -1
        elif edge_type == 'falling':
            step_waveform[int(step_length / 2) :] = -1
    elif step_type == 'erf':
        step_waveform = scipy.special.erf(np.arange(-step_length/2, step_length/2)/scale)
        if edge_type == 'falling':
            step_waveform *= -1
    
    step_waveform = np.interp(x=np.arange(0, step_length - 1, refinement),xp=np.arange(step_length),fp=step_waveform,)
    # find edges
    xcorr = np.apply_along_axis(np.correlate, 1, refined_data, v=step_waveform, mode='valid')
    edge_position = np.argmax(xcorr, axis=1).astype(float) * refinement
    xcorr_amplitude = np.amax(xcorr, axis=1)

    # correct edge_position for step_length
    edge_position += np.floor(step_length / 2)
    #return {'edge_pos': edge_position, 'xcorr': xcorr, 'xcorr_ampl': xcorr_amplitude}
    return edge_position, xcorr_amplitude

def find_fall(a,frac=0.5):
    a = -a
    a = a -np.max(a)
    imx = np.argmin(a)
    print('imx',imx)
    try:
        riseTime = np.interp(frac*a[imx],a[imx:],np.arange(len(a[imx:])))
        riseBin = int(imx+np.round(riseTime))
    except:
        print('interpolation failed')
        riseTime = np.NAN
        riseBin = np.NAN
    return riseBin, -a[imx]

def find_rise(a,frac=0.5):
    a = a -np.max(a)
    imx = np.argmin(a)
    print('imx',imx)
    try:
        riseTime = np.interp(frac*a[imx],a[imx:],np.arange(len(a[imx:])))
        riseBin = int(imx+np.round(riseTime))
    except:
        print('interpolation failed')
        riseTime = np.NAN
        riseBin = np.NAN
    return riseBin, -a[imx]

def find_hm(a):
    a = a-np.min(a)
    imx = np.argmax(a)
    print('imx',imx)
    #print(imx, np.argmin(a[mxpts[0]:mxpts[1]]-bg)+mxpts[0])
    try:
        riseTime = np.interp(0.5*a[imx],a,np.arange(len(a)))
        riseBin = np.round(riseTime)
    except:
        print('interpolation failed')
        riseTime = np.NAN
        riseBin = np.NAN
    return riseBin, a[imx]


##########################################################################
