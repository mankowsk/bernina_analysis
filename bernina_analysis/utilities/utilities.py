import pandas as pd
import numpy as np
import dask.array as da
import sys
# sys.path.insert(0,'/sf/bernina/config/src/python/escape_dev/')
import escape.parse.swissfel as sf
import h5py
import escape as esc
import scipy
from pathlib import Path

### DATA CORRECTION AND MANIPULATION FUNCTIONS FOR ESCAPE ARRAYS ###
def mod_pid(ea,n):
    index = ea.index +n
    data = ea.data
    step_lengths = ea.scan.step_lengths
    parameter = ea.scan.parameter
    return esc.Array(data=data, index=index, step_lengths=step_lengths, parameter=parameter)

def correct_ioxos_shape(ea):
    if len(ea.shape)==2:
        return esc.Array(data=ea.data.T[0], index = ea.index, parameter = ea.scan.parameter, step_lengths = ea.scan.step_lengths)
    else:
        return ea

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


### TIMETOOL ANALYSIS AND DATA ANALYSIS SCRIPTS ###

def refine_max(corr):
    poly = np.array([np.polyfit(np.arange(-10,10),y[np.argmax(y)-10:np.argmax(y)+10], 5) for y in corr.data])
    roots = np.array(  [np.roots(np.polyder(pol,1)) for pol in poly]  )
    deltas = np.array(  [np.real(root[np.argmin(abs(root))]) for root in roots])
    return deltas

def analyse_edge_correlation(tt_sig, ratio_av=None, roi=[650,1350], debug = False, px_smooth=10):
    #"""tt_sig: dictionary with keys 'on' and 'off', which hold 1d escape arrays of timetool traces with and without overlapped laser and x-ray pulses. The script correlates an average of 100 edges with the individual shots and returns two escape arrays, holding the position of the edge in px and the amplitude of the correlation peak."""
    tt_sig['off_sm'] = tt_sig['off'].map_index_blocks(scipy.ndimage.uniform_filter, size=(10,px_smooth))
    tt_sig['on_sm'] = tt_sig['on'].map_index_blocks(scipy.ndimage.uniform_filter, size=(1,px_smooth))
    idx = np.digitize(tt_sig['on'].index, tt_sig['off'].index[:-1]-0.5)
    tt_ratio_sm = esc.Array(index=tt_sig['on_sm'].index, data=tt_sig['on_sm'].data/tt_sig['off_sm'].data[idx]-1, step_lengths=tt_sig['on_sm'].scan.step_lengths, parameter =tt_sig['on_sm'].scan.parameter )
    corr = tt_ratio_sm.map_index_blocks(scipy.ndimage.correlate1d, ratio_av[roi[0]:roi[1]], axis=1, dtype=np.float64)
    corr = corr.compute()
    corr_amp = np.max(corr.data, axis=1)
    corr_pos = np.argmax(corr.data, axis=1)
    deltas = refine_max(corr)
    corr_pos = corr_pos + deltas
    corr_pos_ea = create_ea_from_data(ea=tt_sig['on'], data=corr_pos)
    corr_amp_ea = create_ea_from_data(ea=tt_sig['on'], data=corr_amp)
    if debug:
        return corr_pos_ea, corr_amp_ea, tt_ratio_sm, ratio_av, corr
    else:
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

def load_tt_data(runno, base_dir='/sf/bernina/config/exp/21a_trigo/res/scan_info/'):
    """loads the tt data from a json file"""
    json_dir = Path(f'{base_dir}')
    json_file = list(json_dir.glob(f'run{runno:04d}*'))[0].as_posix()
    data = esc.swissfel.parse_scan(json_file, exclude_from_files=['. .', 'JF', 'CAM'])
    info = sf.readScanEcoJson_v01(json_file)
    pos = np.squeeze(info[0]['scan_values'])
    return pos, data

def prep_tt_data(data, pid_offset=0, cam="M5"):
    """takes the data after parsing with load_tt_data, matched pids and sorts in on and off for analysis"""
    tt_sig = data[f'SARES20-CAMS142-{cam}.roi_signal_x_profile']
    i0 = (data['SLAAR21-LSCP1-FNS:CH4:VAL_GET']+data['SLAAR21-LSCP1-FNS:CH5:VAL_GET']+data['SLAAR21-LSCP1-FNS:CH6:VAL_GET']+data['SLAAR21-LSCP1-FNS:CH7:VAL_GET']).compute()
    evts=data['SAR-CVME-TIFALL5:EvtSet'].compute()
    tt_sig = mod_pid(tt_sig, pid_offset)
    tt_sig, i0, evts = esc.match_arrays (tt_sig,i0,evts)
    i0, tt_sig = on_off([i0, tt_sig], evts)
    return tt_sig, i0, evts

def erf_edge(pts, step_width):
    return scipy.special.erf(np.linspace(start=-pts*2/step_width, stop=pts*2/step_width, num=pts))

def refine_reference(data,pos,resolution=1, width=200):
    """refining the reference signal based on many example datasets (data) and given positions found before"""
    xb = np.arange(len(data[0]))
    xd = xb-np.asarray(pos).ravel()[:,None]
    xd_mn = np.max(xd[:,0])
    xd_mn -= xd_mn%resolution
    xb_mn = xd_mn - resolution/2
    xd_mx = np.min(xd[:,-1])
    xd_mx += (resolution - xd_mx%resolution)
    xb_mx = xd_mx + resolution/2
    xr = np.arange(xd_mn,xd_mx+resolution,resolution)
    xb = np.arange(xb_mn,xb_mx+resolution,resolution)
    bns = np.digitize(xd,xb)
    yr = np.bincount(bns.ravel(),weights=data.ravel())/np.bincount(bns.ravel())
    return xr,yr


def do_fft(t,tr,lm=None,lx=None,plot=None,fname=None,pad=None):

    if lm is not None and lx is not None:
        lm = np.argmin(abs(t-lm));
        lx = np.argmin(abs(t-lx))
        tr=tr[lm:lx]
        t=t[lm:lx]
    if pad is not None:
        N = t.size
        te = t[1]-t[0]
        t = np.arange(N+pad)*te+t[0]
        trp = np.zeros(N+pad)
        trp[:N]=tr
        tr=trp

    N = t.size
    T = t[N-1]-t[0]
    te = t[1]-t[0]
    fe = 1.0/te
    tfd=np.fft.fft(tr)/N
    ampl =np.absolute(tfd)
    freq=np.arange(N)*1.0/T
    if plot:

        fig,ax = plt.subplots(1,1,num='FFT')
        ax.plot(freq[1:300],ampl[1:300]*100,'-o',linewidth=2)
        ax.set_xlabel("Frequency / THz")
        ax.set_ylabel("Amplitude x 100")
        ax.grid()
        ax.legend(frameon=True)
        ax.set_xlim(1,10)
    return freq,ampl

##########################################################################

def binning(x, data_arrays, bin_size=None):
    """
    x: numpy array or list of the old x positions
    data_arrays: list of numpy arrays with data arrays to be binned
    This functionn rebins the data from x_old into x_new. 
    If no bin_size is chose, it defaults to the difference between 
    the first elements of the old x axis"""
    if bin_size is None:
    	bin_size = np.abs(x[1]-x[0])
    bins = np.arange(np.nanmin(x),np.nanmax(x)-bin_size/2,bin_size)+bin_size/2
    pos = np.arange(np.nanmin(x),np.nanmax(x)+bin_size/2,bin_size)
    assigned_bin = np.digitize(x,bins)
    binned_data_arrays = [np.array([np.nansum(dat[(assigned_bin == i)],axis=0) for i in range(len(pos))]) for dat in data_arrays]
    return pos, binned_data_arrays
   
####### elog posting ########

import elog as _elog_ha
from getpass import getuser as _getuser
from getpass import getpass as _getpass


class Elog:
    def __init__(self, url="https://elog-gfa.psi.ch/Bernina", screenshot_directory="", **kwargs):
        self._log, self.user = self.getDefaultElogInstance(url, **kwargs)
        self.read = self._log.read

    def getDefaultElogInstance(self, url, **kwargs):
        from pathlib import Path
        home = str(Path.home())
        if not ("user" in kwargs.keys()):
            kwargs.update(dict(user=_getuser()))
        if not ("password" in kwargs.keys()):
            try:
                with open(os.path.join(home, ".elog_psi"), "r") as f:
                    _pw = f.read().strip()
            except:
                print("Enter elog password for user: %s" % kwargs["user"])
                _pw = _getpass()
        kwargs.update(dict(password=_pw))
        return _elog_ha.open(url, **kwargs), kwargs["user"]

    def post(self, *args, Title=None, Author=None, **kwargs):
        """"""
        if not Author:
            Author = self.user
        return self._log.post(*args, Title=Title, Author=Author, **kwargs)
    
    def attach(self, msg_id, fname, *args, **kwargs):
        message, meta, attach = self._log.read(msg_id)
        meta.pop('$@MID@$')
        meta.update({"attachments": [fname]})
        self.post(message, msg_id=msg_id, **meta )        
        print(f'Attaching to message id {msg_id} with attachment {fname}')
        self.post(message, msg_id=msg_id, **kwargs)

class Container:
    def __init__(self, df, name=''):
        self._cols = df.columns
        self._top_level_name = name
        self._df = df
        self.__dir__()
    
    def _slice_df(self):
        next_level_names = self._get_next_level_names()
        try:
            if len(next_level_names)==0:
                sdf = self._df[self._top_level_name[:-1]]
            else:
                columns_to_keep = [f'{self._top_level_name}{n}' for n in next_level_names if f'{self._top_level_name}{n}' in self._cols]
                sdf = self._df[columns_to_keep]
        except:
            sdf = pd.DataFrame(columns=next_level_names)
        return sdf
    
    def _get_next_level_names(self):
        if len(self._top_level_name) == 0:
            next_level_names = np.unique(np.array([n.split('.')[0] for n in self._cols]))
        else:
            next_level_names = np.unique(np.array([n.split(self._top_level_name)[1].split('.')[0] for n in self._cols if len(n.split(self._top_level_name))>1]))
        return next_level_names

    def _create_first_level_container(self, names):
        for n in names:
            self.__dict__[n]=Container(self._df, name=self._top_level_name+n+'.')

    def to_dataframe(self):
        return self._slice_df()
    
    def __dir__(self):
        next_level_names = self._get_next_level_names()
        to_create = np.array([n for n in next_level_names if not n in self.__dict__.keys()])
        directory = list(next_level_names)
        directory.append('to_dataframe')
        self._create_first_level_container(to_create)
        return directory

    def __repr__(self):
        return self._slice_df().T.__repr__()

    def _repr_html_(self):
        if hasattr(self._slice_df(), '_repr_html_'):
            return self._slice_df().T._repr_html_()
        else:
            return None

    def __getitem__(self, key):
        return self._slice_df().loc[key]