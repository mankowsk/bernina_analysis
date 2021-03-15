from pathlib import Path
import numpy as np
import escape.parse.swissfel as sf
import escape
import os
import time
import json
from esc_hundle import JF_Handler_Escape
import datastorage as dsg
import time
import dask.array as da
import h5py
from escape import Array as ea
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle    
import escape as esc
import jungfrau_utils as ju

plt.ion()

class Data_Reduction():

    def __init__(
        self,
        cfg_file,
    ):

        """load config file"""
        self.cfg_file = cfg_file
        self.cfg= {}
        self.update_cfg_from_file()
        """global parameters"""
        self.dir_save = '/sf/bernina/data/{}/work/analysis/{}/'.format(self.cfg['Exp_config']['pgroup'] ,self.cfg['Exp_config']['save_folder'])
        self.channels = {}
        self.ds = {}

    def update_cfg_from_file(self, cfg_file=None):
        if not cfg_file:
            cfg_file=self.cfg_file
        if type(cfg_file) == dict:
            self.cfg = cfg_file
        else:
            with Path(cfg_file).open(mode='r') as f: self.cfg=json.load(f)
            """convert string True, False and None into booleans"""
            for key, value in self.cfg['Exp_config'].items():
                if value == "True":
                    self.cfg['Exp_config'][key]=True
                if value == "False":
                    self.cfg['Exp_config'][key]=False
                if value == "None":
                    self.cfg['Exp_config'][key]=None
            for jf, cfg in self.cfg['JFs'].items():
                for key, value in cfg.items():
                    if value == "True":
                        self.cfg['JFs'][jf][key]=True
                    if value == "False":
                        self.cfg['JFs'][jf][key]=False
                    if value == "None":
                        self.cfg['JFs'][jf][key]=None

        
    def _setup_data_handlers(self, scan_info):
        jf_handlers = {}
        for jf, jf_cfg in self.cfg['JFs'].items():
            pedestal_file = self.cfg['Exp_config']['pedestal_file']+f".{jf_cfg['id']}.res.h5"
            gain_file = self.cfg['Exp_config']['gain_file']
            jf_file = [file for file in scan_info['scan_files'][0] if jf_cfg['id'] in file][0]
            jf_handlers[jf] = JF_Handler_Escape(
                    file_path = jf_file, 
                    gain_file = gain_file, 
                    pedestal_file = pedestal_file,
                    jf_id=jf_cfg['id']
                    )
        return jf_handlers


    def parse_scan(self, runno, exclude_files):
        pgroup = self.cfg['Exp_config']['pgroup']
        json_dir = Path(f'/sf/bernina/data/{pgroup}/res/scan_info/')
        json_file = list(json_dir.glob(f'run{runno:04}*'))[0]
        print(f'Parsing {json_file}')
        data = sf.parseScanEco_v01(json_file,createEscArrays=True, memlimit_mD_MB=500,exclude_from_files= exclude_files)
        info, info_path = sf.readScanEcoJson_v01(json_file)
        return data, info
    
    def _apply_threshold(self, data, threshold):
        data[data<threshold] = 0
        return data
    
    def _apply_energy_limits(self, data, energy_limits):
        data[data<np.min(energy_limits)] = 0
        data[data>np.max(energy_limits)] = 0
        return data

    def _setup_ds(self, fname):
        ds = dsg.DataStorage(filename = fname)
        print('Setting up datastorage file')
        ds['config'] = self.cfg
        ds['channels'] = {}
        self.ds = ds
        ds.save()
        return ds

    def _setup_filename(self, runno, name, overwrite):
        fname = '{}run_{}.h5'.format(self.dir_save, runno)
        if name: 
            fname = '{}run_{}_{}.h5'.format(self.dir_save, runno, name)
        if os.path.isfile(fname):
            print('The file {} already exists'.format(fname))
            if overwrite == 'question':
                mes = input('Do you want to remove (y,n) or rename (type new name):')
                if mes == 'y':
                    mes = True
                elif mes == 'n':
                    mes = False
            else: 
                mes = overwrite
                print(mes)
            if mes:
                os.remove(fname)
                ds = self._setup_ds(fname)
                ds.save()
            elif not mes:
                ds = dsg.DataStorage(filename = fname)
            else:
                fname = '{}run_{}_{}.h5'.format(self.dir_save, runno, mes)
                ds = self._setup_ds(fname)
                ds.save()
        else:
            ds = self._setup_ds(fname)
            ds.save()
        print(fname)
        return ds, fname


    def _setup_data_reduction_rois(self, jf_handlers, data):
        jfs_rois={}
        for jf, jf_cfg in self.cfg['JFs'].items():
            try:
                jf_data = data[jf_cfg['id']]
                jf_data = jf_data.map_index_blocks(jf_handlers[jf].handler.process, geometry=True, gap_pixels=True, dtype=np.float32)
                threshold = jf_cfg['threshold']
                if not threshold == None:
                    jf_data = jf_data.map_index_blocks(self._apply_threshold, threshold=threshold,dtype=np.float32)
            
                energy_limits = jf_cfg['energy_limits']
                if not energy_limits == None:
                    print('Applying the super cool new energy limits!')
                    jf_data = jf_data.map_index_blocks(self._apply_energy_limits, energy_limits=energy_limits,dtype=np.float32)
                jfs_rois[jf]={'rois':{}, 'rois_img':{}}
                for roi, roi_rng in jf_cfg['rois'].items():
                    r = roi_rng
                    jfs_rois[jf]['rois'][roi] = jf_data[:,r[0]:r[1],r[2]:r[3]].nansum(axis=(1,2))
                for roi, roi_rng in jf_cfg['rois_img'].items():
                    r=roi_rng
                    jfs_rois[jf]['rois_img'][roi] = jf_data[:,r[0]:r[1],r[2]:r[3]]
            except KeyError:
                print('JF {} with id ({}) not in parsed data'.format(jf,jf_cfg['id']))

        return jfs_rois

    def showdata(self, runno, step,exclude_files='CAMERA', jf=None, maxshots = None, clim=(0,20), show_hist=True):
        if not jf:
            jf = 'JFscatt'
        jf_cfg = self.cfg['JFs'][jf]
        pgroup = self.cfg['Exp_config']['pgroup']
        pedestal_file = self.cfg['Exp_config']['pedestal_file']+f".{jf_cfg['id']}.res.h5"

        json_dir = Path(f'/sf/bernina/data/{pgroup}/res/scan_info/')
        print(json_dir)
        json_file = list(json_dir.glob(f'run{runno:04}*'))[0]

        p = Path(json_file)
        with p.open(mode="r") as f:
            s = json.load(f)
            f = [f for f in s['scan_files'][step] if jf_cfg['id'] in f][0]
        with ju.File(file_path= f, pedestal_file=pedestal_file, geometry=True, gap_pixels=True) as juf: 
            if maxshots:
                imgs_ju = juf[:maxshots]
            else:
                imgs_ju = juf[:]
            threshold = jf_cfg['threshold']
            if not threshold == None:
                imgs_ju[imgs_ju<threshold]=0
            energy_limits = jf_cfg['energy_limits']
            if not energy_limits == None:
                imgs_ju[imgs_ju<np.min(energy_limits)]=0
                imgs_ju[imgs_ju>np.max(energy_limits)]=0
            nshots = len(imgs_ju)
            img = np.nansum(imgs_ju,axis=0)
            def hist_only_y(*args,**kwargs):
                y, bins = np.histogram(*args,**kwargs)
                return y
            bins = np.linspace(-5,50,400)
            hists_vertical = np.apply_along_axis(hist_only_y, 1,np.hstack(imgs_ju), bins=bins)           
            hist_av = np.histogram(imgs_ju, bins=bins)
            
        fig = plt.figure(num='Run_{}, Step_{}. Nbr_shots = {}'.format(runno,step, nshots), figsize=(9,9))
        ax0= fig.add_subplot(1,2,1)
        ax0.imshow(img,clim=clim,origin='lower')
        ax1=[]
        
        rois = jf_cfg['rois']
        rois_img = jf_cfg['rois_img']

        print(rois)
        print(rois_img)
        for i in range(1,len(rois)+len(rois_img)+1):
            ax1.append(fig.add_subplot(len(rois)+len(rois_img),2,2*i))
           
        for i,(r,r_rng)  in enumerate(rois.items()):
            ax1[i].imshow(self._mkroi(img, r_rng),clim=clim)
            ax1[i].set_title('{}'.format(r))
            rr =r_rng
            ax0.add_patch(Rectangle((rr[2], rr[0]), abs(rr[2]-rr[3]),abs(rr[1]-rr[0]),fill=None, 
                          alpha=1,linewidth=2, edgecolor='r'))    
                                              
        for i,(r,r_rng)  in enumerate(rois_img.items()):
            i +=len(rois.keys())
            ax1[i].imshow(self._mkroi(img, r_rng),clim=clim)
            ax1[i].set_title('Img {}'.format(r))
            rr =r_rng
            ax0.add_patch(Rectangle((rr[2], rr[0]), abs(rr[2]-rr[3]),abs(rr[1]-rr[0]),fill=None, 
                          alpha=1,linewidth=2, edgecolor='g'))    
        plt.tight_layout()

        return img, fig, bins, hist_av, hists_vertical

    def _mkroi(self, dat, r):
        return dat[r[0]:r[1],r[2]:r[3]]

    def _update_channels(self):
        jf_ids = [self.cfg['JFs'][jf]['id'] for jf in self.cfg['JFs'].keys()]
        bs_ids =  [self.cfg['BSs'][bs]['id'] for bs in self.cfg['BSs'].keys()]
        for jf_id in jf_ids:
            try:
                self.channels.remove(jf_id)
            except:
                print('{} not in parsed data'.format(jf_id))

        self.cfg['other']=self.channels

    def data_reduction(self, runno, exclude_files=['IMAGE'],name=False, overwrite='question'):
        data, scan_info = self.parse_scan(runno,exclude_files)
        poss = np.squeeze(scan_info['scan_values'])
        self.channels = list(data.keys())
        self._update_channels()

        jf_handlers = self._setup_data_handlers(scan_info)
        ds, fname = self._setup_filename(runno, name, overwrite)
        self.ds = ds

        with h5py.File(Path(fname).expanduser(), 'a') as f:
            chsgrp = f.require_group('channels')

            for bs in self.channels:
                if 'processing' in bs:
                    continue
                if 'FPICTURE' in bs:
                    continue
                print('Computing {}'.format(bs))
                group = '/data/'+bs
                dummy = data[bs].index
                data[bs].store(f, group)

                if not bs in chsgrp.keys():
                    chsgrp.update({bs:group})



            jfs_rois = self._setup_data_reduction_rois(jf_handlers, data)
            if len(jfs_rois.keys()) > 0:
                print('Computing JFs')

            for jf, jf_rois in jfs_rois.items():
                rois_comp = []
                for rois, roi_datas in jf_rois.items():
                    for roi, roi_data in roi_datas.items():
                        group = '/data/'+jf+'/'+rois+'/'+roi
                        if not jf+'_'+rois+'_'+roi in chsgrp.keys():
                            chsgrp.update({jf+'_'+rois+'_'+roi:group})
                        #roi_data.store(f,group)
                        roi_data.set_h5_storage(f,group)
                        rois_comp.append(roi_data)
                print('Computing JF {}'.format(jf))
                esc.storage.store(rois_comp)

