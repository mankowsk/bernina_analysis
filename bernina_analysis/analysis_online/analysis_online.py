import scipy
from bsread import source
from escape import Array
from collections import deque
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from threading import Thread
from time import sleep
from epics import PV
import os
from ..utilities.utilities import on_off
from ..utilities.utilities import find_fall, find_rise







class TtProcessor:
    def __init__(self,Nshots = 100, step_type='data', direction='rising', step_width=200):
        """
        Nshots:     number of shots acquired before each evaluation
        step_type:  'data' or 'erf'
        direction:  'falling' or 'rising'. 
            'data', the first 100 evaluation is used to extract the reference from the data."""
        #self.feedback = PV('', auto_monitor=True)
        self.Nshots = Nshots
        self.roi=None
        self.step_type=step_type
        self.direction=direction
        self.step_width = step_width
        self.corr_pos = deque([],2000)
        self.corr_pos_av = deque([],300)
        self.corr_pos_av_std  = deque([],300)
        self.corr_amp = deque([],2000)
        self.corr_amp_av = deque([],300)
        self.corr_amp_av_std = deque([],300)
        self.feedback = True
        self.tt_sig = np.ndarray((Nshots))
        self.tt_ratio_sm = None
        self.evts = np.ndarray((Nshots))
        self.ids = np.ndarray((Nshots))
        self.ratio_av = None
        self.fig = None
        self._running = True
        self.verbose = 0
        self.accumulator = Thread(target=self.run_continuously)
        self.accumulator.start()



    def stop(self):
        self._running = False

    def run_continuously(self):
        with source(channels=['SARES20-CAMS142-M5.roi_signal_x_profile','SAR-CVME-TIFALL5:EvtSet']) as s:
            counter = 0
            while self._running:

                m = s.receive()
                ix = m.data.pulse_id

                prof = m.data.data['SARES20-CAMS142-M5.roi_signal_x_profile'].value
                if prof is None:
                    continue
                evt = m.data.data['SAR-CVME-TIFALL5:EvtSet'].value
                if evt is None:
                    continue
                try:
                    if (lastgoodix-ix)>1:
                        print(f'missed  {lastgoodix-ix-1} events!')
                except:
                    pass
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
    def analyse_edge_correlation_noea(self, tt_sig, ids, ratio_av=None, roi=[650,1350]):
        tt_sig['off_sm'] = scipy.ndimage.uniform_filter(tt_sig['off'], size=(10,10))
        tt_sig['on_sm'] = scipy.ndimage.uniform_filter(tt_sig['on'], size=(1,10))
        idx = np.digitize(ids['on'], ids['off'][:-1]-0.5)
        tt_ratio_sm = tt_sig['on_sm']/tt_sig['off_sm'][idx]-1
        corr = scipy.ndimage.correlate1d(tt_ratio_sm, ratio_av[roi[0]:roi[1]], axis=1)
        corr_amp = np.max(corr.data, axis=1)
        corr_pos = np.argmax(corr.data, axis=1)
        return corr_pos, corr_amp, tt_ratio_sm

    def evaluate(self):
        if self.verbose ==1:
            print(f'ids {self.ids.shape}')
            print(self.ids)
            print(f'evts {self.evts.shape}')
            print(self.evts[:10,25])
            print(f'tt_sig {self.tt_sig.shape}')
            print(self.tt_sig[:10,25])

        tt_sig, ids = on_off([self.tt_sig, self.ids], self.evts)
        if self.ratio_av is None:
            if self.step_type == 'data':
                if len(ids['off'])==0:
                    print("No delayed shots")
                tt_sig['off_sm'] = scipy.ndimage.uniform_filter(tt_sig['off'], size=(10,10))
                tt_sig['on_sm'] = scipy.ndimage.uniform_filter(tt_sig['on'], size=(1,10))
                self.ratio_av=np.mean(tt_sig['on_sm'][:100],axis=0)/np.mean(tt_sig['off_sm'][:100],axis=0)-1
                if self.direction == 'falling':
                    cen, amp = find_fall(self.ratio_av)
                else:
                    cen, amp = find_rise(self.ratio_av)
                if cen < self.step_width:
                    cen = self.step_width
                elif len(self.ratio_av)-cen < self.step_width:
                    cen = len(self.ratio_av)- self.step_width
                self.roi = [int(cen-self.step_width), int(cen+self.step_width)]

            elif self.step_type == 'erf':
                pts = len(self.tt_sig[-1])
                self.ratio_av = scipy.special.erf(np.linspace(start=-pts*2/self.step_width, stop=pts*2/self.step_width, num=pts))
                self.roi = [int(pts/2-self.step_width), int(pts/2+self.step_width)]
                if self.direction == 'falling':
                    self.ratio_av = -self.ratio_av



        corr_pos, corr_amp, tt_ratio_sm = self.analyse_edge_correlation_noea(tt_sig, ids, ratio_av=self.ratio_av, roi=self.roi)
        self.tt_ratio_sm = tt_ratio_sm
        print('analysed')
        self.corr_pos.append(corr_pos)
        self.corr_pos_av.append(np.median(corr_pos))
        self.corr_pos_av_std.append(np.std(corr_pos))
        self.corr_amp.append(corr_amp)
        self.corr_amp_av.append(np.median(corr_amp))
        self.corr_amp_av_std.append(np.std(corr_amp))
        #if not self.fig:
        #    self.setup_plot()
        #self.update_plot()
        return

    def setup_plot(self):
        plt.ion()
        self.fig,self.axs = plt.subplots(2,2,num="BSEN drift monitor")
        self.axs[0][0].set_title('Edge position')
        self.axs[0][1].set_title('Corr amplitude')
        self.axs[1][0].set_title('Last ratio')
        self.axs[1][1].set_title('Reference ratio')
        self.axs[0][0].plot(np.asarray(self.corr_pos_av), color='royalblue', label ='edge pos (px)')
        self.axs[0][0].plot(np.asarray(self.corr_pos_av)+np.asarray(self.corr_pos_av_std), color='royalblue', alpha=0.3, label=f'+/- std')
        self.axs[0][0].plot(np.asarray(self.corr_pos_av)-np.asarray(self.corr_pos_av_std), color='royalblue', alpha=0.3)
        self.axs[0][1].plot(np.asarray(self.corr_amp_av), color='seagreen', label='edge amp')
        self.axs[0][1].plot(np.asarray(self.corr_amp_av)+np.asarray(self.corr_amp_av_std), color='seagreen', alpha=0.3, label=f'+/- std')
        self.axs[0][1].plot(np.asarray(self.corr_amp_av)-np.asarray(self.corr_amp_av_std), color='seagreen', alpha=0.3)
        self.axs[0][0].set_ylabel(f'{self.Nshots} shot_av pos (px)')
        self.axs[0][1].set_ylabel(f'{self.Nshots} shot av amp (px)')
        self.axs[0][0].set_xlabel(f'shots $\cdot$ {self.Nshots}')
        self.axs[0][1].set_xlabel(f'shots $\cdot$ {self.Nshots}')
        self.axs[0][0].legend(loc='upper right', frameon=False)
        self.axs[0][1].legend(loc='upper right', frameon=False)

        self.axs[1][1].plot(self.ratio_av, color='royalblue')
        self.axs[1][1].axvline(self.roi[0], color='black', linestyle='dashed', linewidth=1)
        self.axs[1][1].axvline(self.roi[1], color='black', linestyle='dashed', linewidth=1)
        self.axs[1][0].plot(self.tt_ratio_sm[-1], color='royalblue')
        self.axs[1][0].axvline(self.corr_pos[-1][-1], color='red', linestyle='dashed', linewidth=1)
        self.axs[1][0].set_ylabel(f'on/off')
        self.axs[1][1].set_ylabel(f'on/off')
        self.axs[1][0].set_xlabel(f'pixel')
        self.axs[1][1].set_xlabel(f'pixel')
        self.fig.tight_layout()

    def update_ax_data(self, ax, linnos, xs, ys, scale=None):
        if scale is None:
            scale = np.array([True for n in linnos])
        for linno, x, y in zip(linnos, xs, ys):
            line = ax.get_lines()[linno]
            x = np.asarray(x)
            y = np.asarray(y)
            line.set_data(x,y)
        ymin = np.min(np.array([np.min(y) for y in ys])[scale])
        ymax = np.max(np.array([np.max(y) for y in ys])[scale])
        xmin = np.min(np.array([np.min(x) for x in xs])[scale])
        xmax = np.max(np.array([np.max(x) for x in xs])[scale])
        dy = (ymax-ymin)*0.1
        ax.set_ylim(ymin-dy,ymax+dy)
        ax.set_xlim(xmin-0.5, xmax+0.5)
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
        y = self.tt_ratio_sm[-1]
        edgepos =  self.corr_pos[-1][-1]
        self.update_ax_data(self.axs[1][0], [0,1], [x,[edgepos,edgepos]],[y,[0,1]], scale=np.array([True,False]))
        #self.fig.canvas.draw()
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
            self.ani = FuncAnimation(self.fig,self.update_plot, interval=1000)
            #plt.show()




# tt = TtProcessor()
# sleep(5)
# tt.plot_animation()



