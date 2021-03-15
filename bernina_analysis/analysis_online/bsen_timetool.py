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
from ..utilities.utilities import *


plt.ion()



class TtProcessor:
    def __init__(self,Nshots = 100):
        #self.feedback = PV('', auto_monitor=True)
        self.corr_pos = deque([],2000)
        self.corr_pos_av = deque([],300)
        self.corr_pos_av_std  = deque([],300)
        self.corr_amp = deque([],2000)
        self.corr_amp_av = deque([],300)
        self.corr_amp_av_std = deque([],300)
        self.feedback = True
        self.tt_sig = np.ndarray((,Nshots))
        self.evts = np.ndarray((,Nshots))
        self.ids = np.ndarray((,Nshots))
        self.ratio_av = None
        self.fig = None
        self.accumulator = Thread(target=self.run_continuously)
        self.accumulator.start()

    def run_continuously(self):
        with source(channels=['SARES20-CAMS142-M5.roi_signal_x_profile','SAR-CVME-TIFALL5:EvtSet']) as s:
            counter = 0
            while True:
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
                    self.tt_sig = np.ndarray((Nshots,len(prof)))
                    self.evts = np.ndarray((Nshots,len(evt)))
                    self.ids = np.ndarray((Nshots,))

                lastgoodix = ix
                self.tt_sig[counter]=prof
                self.evts[counter]=evt
                self.ids[counter]=ix
                counter = counter+1
                if counter == Nshots:
                    counter = 0
                    self.evaluate()

    def evaluate(self):
        tt_sig_ea = Array(data=self.tt_sig, index=self.ids)
        tt_sig_ea = on_off(tt_sig_ea, self.evts)
        if not self.ratio_av:
            tt_sig['off_sm'] = tt_sig['off'].map_index_blocks(scipy.ndimage.uniform_filter, size=(10,10))
            tt_sig['on_sm'] = tt_sig['on'].map_index_blocks(scipy.ndimage.uniform_filter, size=(1,10))
            self.ratio_av=tt_sig['on_sm'][:100].mean(axis=0).compute()/tt_sig['off_sm'][:100].mean(axis=0).compute()-1
        corr_pos_ea, corr_amp_ea = analyse_edge_correlation(tt_sig, ratio_av=self.ratio_av, roi=[650,1350])
        self.corr_pos.append(corr_pos_ea.data)
        self.corr_pos_av.append(np.median(corr_pos_ea.data))
        self.corr_pos_std.append(np.std(corr_pos_ea.data))
        self.corr_amp.append(corr_amp_ea.data)
        self.corr_amp_av.append(np.median(corr_amp_ea.data))
        self.corr_amp_std.append(np.std(corr_amp_ea.data))
        if not self.fig:
            self.setup_plot()
        self.update_plot()

    def setup_plot(self):
        self.lh_pos = self.axs[0][0].plot(self.corr_pos_av)[0]
        self.lh_corr = self.axs[0][1].plot(self.corr_amp_av)[0]
        #self.lh_pos_hist = self.axs[1][0].hist(self.corr_pos)[0]
        #self.lh_corr_hist = self.axs[1][1].hist(self.corr_amp)[0]

    def update_plot(self,dum):
        self.lh_pos.set_ydata(self.corr_pos)
        self.lh_corr.set_ydata(self.corr_amp)
        #self.lh_pos_hist.set_data(plt.hist(self.corr_pos))
        #self.lh_corr_hist.set_data(plt.hist(self.corr_amp))
        return self.lh_pos

    def plot_animation(self,name='TT online ana',animate=True):
        if len(self.corr_pos)<1:
            print('no signals yet')
            return
        self.setup_plot()
        self.fig,self.axs = plt.subplots(2,2,sharex=True,num=name)
        # self.fig.clf()
        # self.ax = self.fig.add_subplot(111)
        if animate:
            self.ani = FuncAnimation(self.fig,self.update_plot,init_func=self.setup_plot)
            plt.show()




tt = TtProcessor()
# sleep(5)
# tt.plot_animation()



