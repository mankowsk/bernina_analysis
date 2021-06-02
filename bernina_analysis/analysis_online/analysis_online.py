import scipy
from bsread import source
from collections import deque
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from threading import Thread
from time import sleep
from epics import PV
import os
from ..utilities.utilities import on_off, find_fall, find_rise, erf_edge, refine_reference, do_fft
from epics import PV

from simple_pid import PID
from time import sleep
import datetime
from pathlib import Path
from lmfit.models import GaussianModel
from lmfit import Parameters



class TtProcessor:
    def __init__(
        self,
        cam = 'M5',
        stream=False,
        fsppx=-2.62,
        Nshots = 100,
        memory=5,
        Nshots_diag = 1000,
        step_type='erf',
        direction='rising',
        step_width=200,
        smooth = 80,
        roi=[None,None],
        save=False,
        savedir = '/sf/bernina/data/p19125/res/drift_data/'
    ):
        """
        cam:        BS dara address will be appended to
        Nshots:     number of shots acquired before each evaluation
        step_type:  'data' or 'erf'
        direction:  'falling' or 'rising'.
            'data', the first 100 evaluation is used to extract the reference from the data after finding positions with an erf function."""
        #self.feedback = PV('', auto_monitor=True)
        self.djpv = None
        self.stream=stream
        if cam == 'M5':
            try:
                self.djpv = PV('SLAAR21-SPECTT:AT')
            except:
                print('Issue with connecting to DJ pv')
        elif cam == 'M1':
            try:
                self.djpv = PV('SLAAR21-SPATTT:AT')
            except:
                print('Issue with connecting to DJ pv')
        try:
            self.feedback_control_pv = PV('SLAAR21-LMOT-M523:MOTOR_1')
        except:
            print('Issue with connecting to feedabck')
        self.cam=cam
        self.fsppx = fsppx
        self.Nshots = Nshots
        self.Nshots_diag = Nshots_diag
        self.roi=roi
        self.edge_roi=None
        self.smooth=smooth
        self.save = save
        self.savedir = savedir + f'{self.cam}/'
        self.memory = memory
        self.step_type=step_type
        self.direction=direction
        self.step_width = step_width
        self.pid=deque([],memory)
        self.corr_pos = deque([],memory)
        self.corr_pos_av = deque([],memory)
        self.corr_pos_av_std  = deque([],memory)
        self.corr_amp = deque([],memory)
        self.corr_amp_av = deque([],memory)
        self.corr_amp_av_std = deque([],memory)
        self.feedback = True
        self.tt_sig = np.ndarray((Nshots))
        self.tt_ratio_sm = None
        self.evts = np.ndarray((Nshots))
        self.ids = np.ndarray((Nshots))
        self.edge = None
        self.tt_pumped = None
        self.fig = None
        self._running = True
        self.verbose = 0
        self.counter_glob = 0
        self.pid_1 = 0
        self.pid_2 = 0
        self.accumulator = Thread(target=self.run_continuously)
        self.accumulator.start()
        self._gaussmod = GaussianModel()


    def stop(self):
        self._running = False

    def save_files(self):
        time = datetime.datetime.now().strftime("%H_%M_%S")
        date = datetime.datetime.now().strftime("%Y_%m_%d")

        file_name1 = f'{self.savedir}/{date}/av/{time}_{int(self.Nshots)}av_{int(self.pid_1)}.npy'
        file_name2 = f'{self.savedir}/{date}/single/{time}_{int(self.pid_1)}.npy'
        data_dir = Path(os.path.dirname(file_name1))
        if not data_dir.exists():
            print(f"Path {data_dir.absolute().as_posix()} does not exist, will try to create it...")
            data_dir.mkdir(parents=True)
            print(f"Tried to create {data_dir.absolute().as_posix()}")
            data_dir.chmod(0o775)
            print(f"Tried to change permissions to 775")
        data_dir = Path(os.path.dirname(file_name2))
        if not data_dir.exists():
            print(f"Path {data_dir.absolute().as_posix()} does not exist, will try to create it...")
            data_dir.mkdir(parents=True)
            print(f"Tried to create {data_dir.absolute().as_posix()}")
            data_dir.chmod(0o775)
            print(f"Tried to change permissions to 775")

        np.save(file_name1,[self.corr_pos_av, self.corr_pos_av_std, self.corr_amp_av, self.corr_amp_av_std])
        np.save(file_name2,[np.hstack(self.pid), np.hstack(self.corr_pos), np.hstack(self.corr_amp)])


    def run_continuously(self):
        with source(channels=[f'SARES20-CAMS142-{self.cam}.roi_signal_x_profile','SAR-CVME-TIFALL5:EvtSet']) as s:
            counter = 0
            while self._running:

                m = s.receive()
                ix = m.data.pulse_id

                prof = m.data.data[f'SARES20-CAMS142-{self.cam}.roi_signal_x_profile'].value
                if prof is None:
                    continue
                evt = m.data.data['SAR-CVME-TIFALL5:EvtSet'].value
                if evt is None:
                    continue
                if counter ==0:
                    self.tt_sig = np.ndarray((self.Nshots,len(prof)))
                    self.evts = np.ndarray((self.Nshots,len(evt)))
                    self.ids = np.ndarray((self.Nshots,))

                lastgoodix = ix
                self.tt_sig[counter]=prof
                self.evts[counter]=evt
                self.ids[counter]=ix
                counter = counter+1
                if counter == self.Nshots:
                    counter = 0
                    self.evaluate()
                    continue
    def analyse_edge_correlation_noea(self, tt_sig, ids, edge=None, roi=[650,1350], smooth=80):
        tt_sig['off_sm'] = scipy.ndimage.uniform_filter(tt_sig['off'], size=(10,smooth))
        tt_sig['on_sm'] = scipy.ndimage.uniform_filter(tt_sig['on'], size=(1,smooth))
        idx = np.digitize(ids['on'], ids['off'][:-1]-0.5)
        tt_ratio_sm = tt_sig['on_sm']/tt_sig['off_sm'][idx]-1
        corr = scipy.ndimage.correlate1d(tt_ratio_sm, edge[roi[0]:roi[1]], axis=1)
        corr_amp = np.max(corr.data, axis=1)
        corr_pos = np.argmax(corr.data, axis=1)
        return corr_pos, corr_amp, tt_ratio_sm

    def take_pumped_background(self):
        self.tt_pumped = np.mean(self.tt_ratio_sm, axis=0)
        self.tt_pumped = self.tt_pumped/np.max(abs(self.tt_pumped))
        self.roi = [len(self.tt_pumped)-find_fall(abs(self.tt_pumped[::-1]),0.2)[0], find_fall(abs(self.tt_pumped),0.2)[0]]

    def evaluate(self):

        if self.verbose ==1:
            print(f'ids {self.ids.shape}')
            print(self.ids)
            print(f'evts {self.evts.shape}')
            print(self.evts[:10,25])
            print(f'tt_sig {self.tt_sig.shape}')
            print(self.tt_sig[:10,25])

        tt_sig, ids = on_off([self.tt_sig[:,self.roi[0]:self.roi[1]], self.ids], self.evts)
        if len(ids['off']) == 0:
            print('No Delayed Shots')
            return
        elif len(ids['on']) == 0:
            print('No Pumped shots')
            return
        if self.counter_glob ==0:
            self.pid_1 = ids['on'][0]
        if self.edge is None:
            if self.step_type == 'data':
                pts = len(self.tt_sig[-1])
                self.edge = erf_edge(pts, self.step_width)
                self.edge_roi = [int(pts/2-self.step_width/2), int(pts/2+self.step_width/2)]
                if self.direction == 'falling':
                    self.edge = -self.edge
                corr_pos, corr_amp, tt_ratio_sm = self.analyse_edge_correlation_noea(tt_sig, ids, edge=self.edge, roi=self.edge_roi, smooth=self.smooth)
                xr, yr = refine_reference(tt_ratio_sm,corr_pos,resolution=1)
                if self.direction == 'rising':
                    cen, amp = find_rise(yr)
                elif self.direction == 'falling':
                    cen, amp = find_fall(yr)
                if cen < self.step_width:
                    cen = self.step_width
                elif len(yr)-cen < self.step_width:
                    cen = len(yr)- self.step_width
                yr = yr[int(cen-self.step_width/2):int(cen+self.step_width/2)]
                self.edge_roi = [None,None]
                self.edge=yr

            elif self.step_type == 'erf':
                pts = len(self.tt_sig[-1])
                self.edge = erf_edge(pts, self.step_width)
                self.edge_roi = [int(pts/2-self.step_width/2), int(pts/2+self.step_width/2)]
                if self.direction == 'falling':
                    self.edge = -self.edge

        corr_pos, corr_amp, tt_ratio_sm = self.analyse_edge_correlation_noea(tt_sig, ids, edge=self.edge, roi=self.edge_roi, smooth=self.smooth)

        if self.tt_pumped is not None:
            tt_ratio_sm = tt_ratio_sm/self.tt_pumped[None,self.roi[0]:self.roi[1]]
        self.tt_ratio_sm = tt_ratio_sm

        self.pid.append(ids['on'])
        self.corr_pos.append(corr_pos)
        self.corr_pos_av.append(np.median(corr_pos))
        self.corr_pos_av_std.append(np.std(corr_pos))
        self.corr_amp.append(corr_amp)
        self.corr_amp_av.append(np.median(corr_amp))
        self.corr_amp_av_std.append(np.std(corr_amp))
        self.counter_glob = self.counter_glob +1
        if self.stream:
            if self.djpv:
                self.djpv.put(self.corr_pos_av[-1]*self.fsppx)
        if self.counter_glob ==self.memory:
            self.pid_2 = ids['on'][-1]
            self.counter_glob = 0

            if self.save:
                self.save_files()

        return

    def setup_feed_back(self,setpoint=1300,output_limits=[]):
        self.fb_pid = PID(1,0,0,setpoint=setpoint,output_limits=output_limits,sample_time=1,proportional_on_measurement=True)

    def feed_back(self,value):
        set_val = self.fb_pid(value)
        sleep(5)
        self.feedback_control_pv.put(set_val)

    def diag_hist_fft(self, data, bin_width=1, nbins=None):
        dat = np.hstack(data)[-self.Nshots_diag:]
        pid = np.hstack(self.pid)[-self.Nshots_diag:]
        pid = pid - pid[0]

        freq, amp = do_fft(pid,dat)
        freq = freq*100
        clip = int(len(freq)/2)
        freq = freq[1:clip]
        amp = amp[1:clip]
        if nbins:
            bin_width= 6*np.std(dat)/nbins
        bins = np.arange(np.median(dat)-3*np.std(dat), np.median(dat)+3*np.std(dat),bin_width)
        std = np.std(dat)

        hist, bins = np.histogram(dat, bins=bins)
        bins = bins[:-1]+(bins[1]-bins[0])/2
        pars = Parameters()
        pars.add('center',value=bins[np.argmax(hist)])
        pars.add('amplitude',value=np.max(hist))
        pars.add('sigma',value=np.std(hist))
        bins = bins[hist>0]
        hist=hist[hist>0]

        fit = self._gaussmod.fit(hist, x=bins, params=pars, weights=1/np.sqrt(hist))

        return freq, amp, hist, bins, fit.best_fit, fit.best_values

    def setup_plot(self):
        plt.ion()
        self.fig,self.axs = plt.subplots(4,2,num=f"BSEN drift monitor cam {self.cam}", figsize=(7,12))
        self.fig.suptitle(f'Camera {self.cam}')
        self.axs[0][0].set_title('Edge position')
        self.axs[0][1].set_title('Correlation amplitude')
        self.axs[1][0].set_title('Last ratio')
        self.axs[1][1].set_title('Reference ratio')
        self.axs[2][0].set_title('FFT (edge pos)')
        self.axs[2][1].set_title('FFT (edge amp)')
        self.axs[3][0].set_title(f'Hist (edge pos {self.Nshots_diag} shots)')
        self.axs[3][1].set_title(f'Hist (edge amp {self.Nshots_diag} shots)')

        self.axs[0][0].plot(np.asarray(self.corr_pos_av), color='royalblue', label ='edge pos (px)')
        self.axs[0][0].plot(np.asarray(self.corr_pos_av)+np.asarray(self.corr_pos_av_std), color='royalblue', alpha=0.3, label=f'+/- std')
        self.axs[0][0].plot(np.asarray(self.corr_pos_av)-np.asarray(self.corr_pos_av_std), color='royalblue', alpha=0.3)
        self.axs[0][1].plot(np.asarray(self.corr_amp_av), color='seagreen', label='edge amp')
        self.axs[0][1].plot(np.asarray(self.corr_amp_av)+np.asarray(self.corr_amp_av_std), color='seagreen', alpha=0.3, label=f'+/- std')
        self.axs[0][1].plot(np.asarray(self.corr_amp_av)-np.asarray(self.corr_amp_av_std), color='seagreen', alpha=0.3)
        self.axs[0][0].set_ylabel(f'{self.Nshots} shot av pos (px)')
        self.axs[0][1].set_ylabel(f'{self.Nshots} shot av amp (px)')
        self.axs[0][0].set_xlabel(f'shots / {self.Nshots}')
        self.axs[0][1].set_xlabel(f'shots / {self.Nshots}')
        self.axs[0][0].legend(loc='upper right', frameon=False)
        self.axs[0][1].legend(loc='upper right', frameon=False)

        self.axs[1][1].plot(self.edge, color='royalblue')
        self.axs[1][1].axvline(self.edge_roi[0], color='black', linestyle='dashed', linewidth=1)
        self.axs[1][1].axvline(self.edge_roi[1], color='black', linestyle='dashed', linewidth=1)
        self.axs[1][0].plot(self.tt_ratio_sm[-1]+1, color='royalblue')
        self.axs[1][0].axvline(self.corr_pos[-1][-1]+1, color='red', linestyle='dashed', linewidth=1)
        self.axs[1][0].set_ylabel(f'on/off')
        self.axs[1][1].set_ylabel(f'on/off')
        self.axs[1][0].set_xlabel(f'pixel')
        self.axs[1][1].set_xlabel(f'pixel')

        freq, amp, hist, bins, fit, fit_values = self.diag_hist_fft(self.corr_pos)
        self.axs[2][0].plot(freq, amp, color='royalblue')
        self.axs[3][0].plot(bins, hist, '.', color='royalblue')
        self.axs[3][0].plot(bins, fit, 'k')

        freq, amp, hist, bins, fit, fit_values = self.diag_hist_fft(self.corr_amp, nbins=100)
        self.axs[2][1].plot(freq, amp, color='seagreen')
        self.axs[3][1].plot(bins, hist, '.', color='seagreen')
        self.axs[3][1].plot(bins, fit, 'k')

        self.axs[2][0].set_ylabel(f'FFT')
        self.axs[2][1].set_ylabel(f'FFT')
        self.axs[2][0].set_xlabel(f'Hz')
        self.axs[2][0].set_xlim(0,100)
        self.axs[2][1].set_xlabel(f'Hz')
        self.axs[2][1].set_xlim(0,100)
        self.axs[3][0].set_ylabel(f'Hist')
        self.axs[3][1].set_ylabel(f'Hist')
        self.axs[3][0].set_xlabel(f'pixel')
        self.axs[3][1].set_xlabel(f'correlation amp')

        self.fig.tight_layout()

    def update_ax_data(self, ax, linnos, xs, ys, scale=None, labels=None):
        if scale is None:
            scale = np.array([True for n in linnos])
        lines = []
        for linno, x, y in zip(linnos, xs, ys):
            line = ax.get_lines()[linno]
            x = np.asarray(x)
            y = np.asarray(y)
            line.set_data(x,y)
            lines.append(line)
        ymin = np.min(np.array([np.min(y) for y in ys])[scale])
        ymax = np.max(np.array([np.max(y) for y in ys])[scale])
        xmin = np.min(np.array([np.min(x) for x in xs])[scale])
        xmax = np.max(np.array([np.max(x) for x in xs])[scale])
        dy = (ymax-ymin)*0.1
        ax.set_ylim(ymin-dy,ymax+dy)
        ax.set_xlim(xmin-0.5, xmax+0.5)
        if labels is not None:
            ax.legend(lines, labels, loc='upper left')
        ax.figure.canvas.draw()


    def update_plot(self, frame):
        #plt.clf()
        x = np.arange(len(self.corr_pos_av))
        y = np.array(self.corr_pos_av)
        ystd =  np.array(self.corr_pos_av_std)
        self.update_ax_data(self.axs[0][0], [0,1,2], [x, x, x],[y, y+ystd, y-ystd])
        x = np.arange(len(self.corr_amp_av))
        y = np.array(self.corr_amp_av)
        ystd =  np.array(self.corr_amp_av_std)
        self.update_ax_data(self.axs[0][1], [0,1,2], [x, x, x],[y, y+ystd, y-ystd])
        x = np.arange(len(self.tt_ratio_sm[-1]))
        y = self.tt_ratio_sm[-1]+1
        edgepos =  self.corr_pos[-1][-1]
        self.update_ax_data(self.axs[1][0], [0,1], [x,[edgepos,edgepos]],[y,[0,1]], scale=np.array([True,False]))
        freq, amp, hist, bins, fit, fit_values = self.diag_hist_fft(self.corr_pos)
        self.update_ax_data(self.axs[2][0], [0], [freq],[amp])
        self.update_ax_data(self.axs[3][0], [0,1], [bins, bins],[hist, fit], labels=['data', f'fwhm {fit_values["sigma"]*2.3548:.4} cen {int(fit_values["center"])}'])
        freq, amp, hist, bins, fit, fit_values = self.diag_hist_fft(self.corr_amp, nbins=100)
        self.update_ax_data(self.axs[2][1], [0], [freq],[amp])
        self.update_ax_data(self.axs[3][1], [0,1], [bins, bins],[hist, fit], labels=['data', f'fwhm {fit_values["sigma"]*2.3548:.4} cen {int(fit_values["center"])}'])
        self.fig.canvas.draw()
        #self.lh_pos_hist.set_data(plt.hist(self.corr_pos))
        #self.lh_corr_hist.set_data(plt.hist(self.corr_amp))
        return

    def plot_animation(self,name='TT online ana',animate=True):
        if len(self.corr_pos)<1:
            print('no signals yet')
            return
        self.setup_plot()
        # self.fig.clf()
        # self.ax = self.fig.add_subplot(111)
        if animate:
            self.ani = FuncAnimation(self.fig,self.update_plot, interval=self.Nshots*10)
            #plt.show()




# tt = TtProcessor()
# sleep(5)
# tt.plot_animation()



