from pathlib import Path
import sys
# sys.path.insert(0,'/sf/bernina/config/src/python/escape-fel/')
import numpy as np

import os
import time
import json
import datastorage as dsg
import time
import dask.array as da
import h5py

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


from escape import Array as ea
import escape as esc
import escape.swissfel as sf
import jungfrau_utils as ju
from dask.utils import SerializableLock
parse_scan = sf.parse.parseScanEco_v01
read_scan = sf.parse.readScanEcoJson_v01
plt.ion()

class Data_Reduction():

    def __init__(
        self,
        cfg_file = {},
        no_jfutils=True,
        do_slice=False,
        maxshots=500,
        multi_limits=False,
    ):

        """load config file"""
        self.do_slice=do_slice
        self.maxshots=maxshots
        self.cfg_file = cfg_file
        self.cfg= {}
        self.update_cfg()
        """global parameters"""
        self.dir_save = '/sf/bernina/data/{}/work/{}/'.format(self.cfg['Exp_config']['pgroup'] ,self.cfg['Exp_config']['save_folder'])
        self.channels = {}
        self.ds = {}
        self.no_jfutils=no_jfutils
        self.multi_limits = multi_limits

    def update_cfg(self, cfg_file=None):
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

    def autolocate_closest_pedestal(self, jf, runno):
        jf_cfg = self.cfg['JFs'][jf]
        jf_id = jf_cfg['id']
        pgroup = self.cfg['Exp_config']['pgroup']
        pedestal_path = Path(f'/sf/jungfrau/data/pedestal/{jf_cfg["id"]}/')
        pedestal_files = np.array([[f.lstat().st_mtime, f] for f in pedestal_path.glob('*.h5') if jf_id in f.as_posix()]).T
        pedestal_path2 = Path(f'/sf/bernina/data/{pgroup}/raw/JF_pedestals/')
        pedestal_files2 = np.array([[f.lstat().st_mtime, f] for f in pedestal_path2.glob('*.h5') if jf_id in f.as_posix()]).T
        pedestal_files = np.hstack([pedestal_files, pedestal_files2])
        pedestal_files = np.sort(pedestal_files, axis=1)
        json_dir = Path(f'/sf/bernina/data/{pgroup}/res/scan_info/')
        json_file = Path(list(json_dir.glob(f'run{runno:04}*'))[0])
        with json_file.open(mode="r") as f:
            s = json.load(f)
            f = [f for f in s['scan_files'][0] if jf_cfg['id'] in f][0]
        jf_file_mtime = Path(f).lstat().st_mtime
        idx = np.argmin(abs(pedestal_files[0]-jf_file_mtime))
        if pedestal_files[0][idx]-jf_file_mtime >0 and idx >0:
            idx = idx-1
        pedestal_file = pedestal_files[1][idx].as_posix()
        dt = jf_file_mtime - pedestal_files[0][idx]
        days = dt // (24 * 3600)
        hours = dt % (24 * 3600) // 3600
        print(f'Located closest pedestal file dated back {days} days, {hours} hours: {pedestal_file}')
        return pedestal_file

    def setup_data_handlers(self, scan_info):
        jf_handlers = {}
        for jf, jf_cfg in self.cfg['JFs'].items():
            pedestal_file = jf_cfg['pedestal_file']
            gain_file = jf_cfg['gain_file']
            jf_file = [file for file in scan_info['scan_files'][0] if jf_cfg['id'] in file][0]
            jf_handlers[jf] = ju.EscapeAdapter(
                    file_path = jf_file,
                    gain_file = gain_file,
                    pedestal_file = pedestal_file,
                    )
        return jf_handlers

    def parse_scan(self, runno, exclude_files):
        pgroup = self.cfg['Exp_config']['pgroup']
        json_dir = Path(f'/sf/bernina/data/{pgroup}/res/scan_info/')
        json_file = list(json_dir.glob(f'run{runno:04}*'))[0]
        print(f'Parsing {json_file}')
        #data = sf.parseScanEco_v01(json_file,createEscArrays=True, memlimit_mD_MB=50,exclude_from_files= exclude_files)
        data = parse_scan(json_file,createEscArrays=True)
        info, info_path = read_scan(json_file)
        return data, info

    def parse_step_from_scan(self, runno, step, exclude_files):
        pgroup = self.cfg['Exp_config']['pgroup']
        json_dir = Path(f'/sf/bernina/data/{pgroup}/res/scan_info/')
        json_file = list(json_dir.glob(f'run{runno:04}*'))[0]
        print(f'Parsing {json_file}')
        data = sf.parseScanEco_v01(json_file,createEscArrays=True, memlimit_mD_MB=50,exclude_from_files= exclude_files)
        info, info_path = sf.readScanEcoJson_v01(json_file)
        return data, info

    def apply_gain_pede_np(self, image, G, P, pixel_mask=None, mask_value = np.nan):
        # gain and pedestal correction
        mask = int('0b' + 14 * '1', 2)
        mask2 = int('0b' + 2 * '1', 2)
        gain_mask = np.bitwise_and(np.right_shift(image, 14), mask2)
        data = np.bitwise_and(image, mask)
        m1 = gain_mask == 0
        m2 = gain_mask == 1
        m3 = gain_mask >= 2
        if G is not None:
            g = m1*G[0] + m2*G[1] + m3*G[2]
        else:
            g = np.ones(data.shape, dtype=np.float64)
        if P is not None:
            p = m1*P[0] + m2*P[1] + m3*P[2]
        else:
            p = np.zeros(data.shape, dtype=np.float64)
        res = np.divide(data - p, g)
        if pixel_mask is not None:
		#dv,mv = np.broadcast_arrays(res,pixel_mask) # seems not necessary
            mv = pixel_mask
            if isinstance(image,da.Array):
                mv = da.from_array(mv, chunks=image.chunks[-2:])
            res[mv!=0] = mask_value
            res=da.nan_to_num(res,0)
        return res

    def apply_threshold(self, data, threshold):
        data[data<threshold] = 0
        return data

    def apply_energy_limits(self, data, energy_limits):
        data[data<np.min(energy_limits)] = 0
        data[data>np.max(energy_limits)] = 0
        return data

    def convert_to_photons(self, data, energy, delta_e):
        data[data<energy-delta_e]=0
        data[np.array([ np.all([m,n]) for m,n in zip(np.remainder(abs(data),energy)>delta_e,np.remainder(abs(data),energy)<energy-delta_e)])]=0
        data=np.floor_divide(data+delta_e,energy)
        return data

    def setup_ds(self, fname):
        ds = dsg.DataStorage(filename = fname)
        print('Setting up datastorage file')
        ds['config'] = self.cfg
        ds['channels'] = {}
        self.ds = ds
        ds.save()
        return ds

    def setup_filename(self, runno, name, overwrite):
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
                ds = self.setup_ds(fname)
                ds.save()
            elif not mes:
                ds = dsg.DataStorage(filename = fname)
            else:
                fname = '{}run_{}_{}.h5'.format(self.dir_save, runno, mes)
                ds = self.setup_ds(fname)
                ds.save()
        else:
            ds = self.setup_ds(fname)
            ds.save()
        print(fname)
        return ds, fname

    def setup_rois(self, data):
        jfs_rois={}
        for jf, jf_cfg in self.cfg['JFs'].items():
            pedestal_file = jf_cfg['pedestal_file']
            gain_file = jf_cfg['gain_file']
            with h5py.File(f'/sf/jungfrau/config/gainMaps/{jf_cfg["id"]}/gains.h5','r') as fh:
                gain = fh['gains'][:]
            with h5py.File(pedestal_file,'r') as fh:
                pedestal = fh['gains'][:]
            try:
                jf_data = data[jf_cfg['id']]
                dummy = jf_data.index
                jf_data = jf_data.map_index_blocks(self.apply_gain_pede_np,gain,pedestal,dtype=np.float64)
                threshold = jf_cfg['threshold']
                if not threshold == None:
                    jf_data = jf_data.map_index_blocks(self.apply_threshold, threshold=threshold,dtype=np.float64)
                energy_limits = jf_cfg['energy_limits']
                if not energy_limits == None:
                    print('Applying the super cool new energy limits!')
                    jf_data = jf_data.map_index_blocks(self.apply_energy_limits, energy_limits=energy_limits,dtype=np.float64)
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

    def setup_rois_ju(self, jf_handlers, data):
        jfs_rois={}
        for jf, jf_cfg in self.cfg['JFs'].items():
            try:
                jf_data = data[jf_cfg['id']]
                dummy = jf_data.index
                jf_data = jf_data.map_index_blocks(jf_handlers[jf].handler.process, geometry=False, gap_pixels=False, dtype=np.float64)
                threshold = jf_cfg['threshold']
                if not threshold == None:
                    jf_data = jf_data.map_index_blocks(self.apply_threshold, threshold=threshold,dtype=np.float64)
                energy_limits = jf_cfg['energy_limits']
                if not energy_limits == None:
                    print('Applying the super cool new energy limits!')
                    jf_data = jf_data.map_index_blocks(self.apply_energy_limits, energy_limits=energy_limits,dtype=np.float64)
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

    def setup_rois_multi_limits(self, data):
        jfs_rois={}
        for jf, jf_cfg in self.cfg['JFs'].items():
            pedestal_file = jf_cfg['pedestal_file']
            gain_file = jf_cfg['gain_file']
            with h5py.File(f'/sf/jungfrau/config/gainMaps/{jf_cfg["id"]}/gains.h5','r') as fh:
                gain = fh['gains'][:]
            with h5py.File(pedestal_file,'r') as fh:
                pedestal = fh['gains'][:]
            try:
                jf_data = data[jf_cfg['id']]
                dummy = jf_data.index
                jf_data = jf_data.map_index_blocks(self.apply_gain_pede_np,gain,pedestal,dtype=np.float64)
                threshold = jf_cfg['threshold']
                if not threshold == None:
                    jf_data = jf_data.map_index_blocks(self.apply_threshold, threshold=threshold,dtype=np.float64)
                energy_limits_alldet = jf_cfg['energy_limits']
                print('energy_limits_alldet')
                print(energy_limits_alldet)
                if energy_limits_alldet:
                    print('Applying energy limits to the entire data')
                    jf_data = jf_data.map_index_blocks(self.apply_energy_limits, energy_limits=[energy_limits_alldet],dtype=np.float64)
                jfs_rois[jf]={'rois':{}, 'rois_img':{}}
                for roiname, roidef in jf_cfg['rois'].items():
                    r = roidef['range']
                    roi_tmp = jf_data[:,r[0]:r[1],r[2]:r[3]]
                    energy_limits = roidef['energy_limits']
                    print('energy_limits')
                    print(energy_limits)
                    roi_tmp = roi_tmp.map_index_blocks(self.apply_energy_limits, energy_limits=energy_limits,dtype=np.float64)
                    jfs_rois[jf]['rois'][roiname] = roi_tmp.nansum(axis=(1,2))
                for roi, roi_rng in jf_cfg['rois_img'].items():
                    r=roi_rng
                    jfs_rois[jf]['rois_img'][roi] = jf_data[:,r[0]:r[1],r[2]:r[3]]
            except KeyError:
                print('JF {} with id ({}) not in parsed data'.format(jf,jf_cfg['id']))


        return jfs_rois

    def showdata_noju(self, runno, step,exclude_files, jf=None, maxshots = None, clim=(0,20), hist_lineout = False):
        data, scan_info = self.parse_scan(runno,exclude_files)
        poss = np.squeeze(scan_info['scan_values'])

        jf_handlers = self.setup_data_handlers(scan_info)
        if not jf:
            jf = 'JFscatt'
        jf_cfg = self.cfg['JFs'][jf]
        jf_data = data[jf_cfg['id']].scan[step]
        if maxshots:
            jf_data = jf_data[:maxshots,:,:]
        jf_data = jf_data.map_index_blocks(jf_handlers[jf].handler.process, geometry=False, gap_pixels=False, dtype=np.float32)
        threshold = jf_cfg['threshold']
        if not threshold == None:
            jf_data = jf_data.map_index_blocks(self.apply_threshold, threshold=threshold,dtype=np.float32)
        imgs = jf_data.data.compute()
        data = {}
        img = np.nansum(imgs,axis=0)
        nshots = len(imgs)
        def hist_only_y(*args,**kwargs):
            y, bins = np.histogram(*args,**kwargs)
            return y
        bins = np.linspace(-5,50,400)
        if hist_lineout:
            hists_vertical = np.apply_along_axis(hist_only_y, 1,np.hstack(imgs), bins=bins)
            data['hists_vertical']=hists_vertical
        hist_av = np.histogram(imgs[:100], bins=bins)
        data['hist_av']=hist_av
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
            if type(r_rng)==dict:
                r_rng = r_rng['range']
            ax1[i].imshow(self.mkroi(img, r_rng),clim=clim, origin='lower')
            ax1[i].set_title('{}'.format(r))
            rr =r_rng
            ax0.add_patch(Rectangle((rr[2], rr[0]), abs(rr[2]-rr[3]),abs(rr[1]-rr[0]),fill=None,
                          alpha=1,linewidth=2, edgecolor='r'))

        for i,(r,r_rng)  in enumerate(rois_img.items()):
            i +=len(rois.keys())
            ax1[i].imshow(self.mkroi(img, r_rng),clim=clim, origin='lower')
            ax1[i].set_title('Img {}'.format(r))
            rr =r_rng
            ax0.add_patch(Rectangle((rr[2], rr[0]), abs(rr[2]-rr[3]),abs(rr[1]-rr[0]),fill=None,
                          alpha=1,linewidth=2, edgecolor='g'))
        plt.tight_layout()
        hist = plt.figure('histogram', figsize=(9,3))
        plt.plot(hist_av[1][:-1],hist_av[0])
        plt.xlabel('energy (keV)')
        plt.ylabel('px/bin')
        plt.yscale('log')
        plt.ylim(hist_av[0][-1],np.max(hist_av[0]))
        plt.tight_layout()
        data['img']=img
        data['fig']=fig
        data['hist_bins']=bins
        return data

    def showdata(self, runno, step,exclude_files, maxshots = None, clim=(0,20), jf=None, hist_lineout = False, autolocate_pedestal=True):
        if jf is None:
            jf = list(self.cfg['JFs'].keys())[0]
        if autolocate_pedestal:
            for j in self.cfg['JFs'].keys():
                jfcfg = self.cfg['JFs'][j]
                ped_file = self.autolocate_closest_pedestal(j, runno)
                jfcfg['pedestal_file'] = ped_file
        jf_cfg = self.cfg['JFs'][jf]
        pgroup = self.cfg['Exp_config']['pgroup']
        pedestal_file = jf_cfg['pedestal_file']

        json_dir = Path(f'/sf/bernina/data/{pgroup}/res/scan_info/')
        print(json_dir)
        json_file = list(json_dir.glob(f'run{runno:04}*'))[0]
        data = {}
        p = Path(json_file)
        with p.open(mode="r") as f:
            s = json.load(f)
            f = [f for f in s['scan_files'][step] if jf_cfg['id'] in f][0]
        with ju.File(file_path= f, pedestal_file=pedestal_file, geometry=False, gap_pixels=False) as juf:
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
            if hist_lineout:
                hists_vertical = np.apply_along_axis(hist_only_y, 1,np.hstack(imgs_ju), bins=bins)
                data['hists_vertical']=hists_vertical
            hist_av = np.histogram(imgs_ju[:100], bins=bins)
            data['hist_av']=hist_av
        figname = 'Run_{}, Step_{}. Nbr_shots = {}'.format(runno,step, nshots)
        plt.close(figname)
        fig = plt.figure(num=figname, figsize=(9,9))
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
            if type(r_rng)==dict:
                r_rng = r_rng['range']
            ax1[i].imshow(self.mkroi(img, r_rng),clim=clim)
            ax1[i].set_title('{}'.format(r))
            rr =r_rng
            ax0.add_patch(Rectangle((rr[2], rr[0]), abs(rr[2]-rr[3]),abs(rr[1]-rr[0]),fill=None,
                          alpha=1,linewidth=2, edgecolor='r'))
            ax0.text(rr[2],rr[1],r,color='white')
        for i,(r,r_rng)  in enumerate(rois_img.items()):
            i +=len(rois.keys())
            ax1[i].imshow(self.mkroi(img, r_rng),clim=clim)
            ax1[i].set_title('Img {}'.format(r))
            rr =r_rng
            ax0.add_patch(Rectangle((rr[2], rr[0]), abs(rr[2]-rr[3]),abs(rr[1]-rr[0]),fill=None,
                          alpha=1,linewidth=2, edgecolor='g'))
        plt.tight_layout()
        hist = plt.figure('histogram', figsize=(9,3))
        plt.plot(hist_av[1][:-1],hist_av[0])
        plt.xlabel('energy (keV)')
        plt.ylabel('px/bin')
        plt.yscale('log')
        plt.ylim(hist_av[0][-1],np.max(hist_av[0]))
        plt.tight_layout()
        data['img']=img
        data['fig']=fig
        data['hist_bins']=bins
        return data


    def mkroi(self, dat, r):
        return dat[r[0]:r[1],r[2]:r[3]]


    def update_channels(self):
        jf_ids = [self.cfg['JFs'][jf]['id'] for jf in self.cfg['JFs'].keys()]
        bs_ids =  [self.cfg['BSs'][bs]['id'] for bs in self.cfg['BSs'].keys()]
        for jf_id in jf_ids:
            try:
                self.channels.remove(jf_id)
            except:
                print('{} not in parsed data'.format(jf_id))

        self.cfg['other']=self.channels

    def data_reduction(self, runno, exclude_files=['IMAGE'],name=False, overwrite='question', autolocate_pedestal = True):
        if autolocate_pedestal:
            print(self.cfg['JFs'])
            for jf in self.cfg['JFs'].keys():
                jfcfg = self.cfg['JFs'][jf]
                ped_file = self.autolocate_closest_pedestal(jf, runno)
                jfcfg['pedestal_file'] = ped_file
        data, scan_info = self.parse_scan(runno,exclude_files)
        poss = np.squeeze(scan_info['scan_values'])

        if 'BS_channels' in self.cfg.keys():
            self.channels = self.cfg['BS_channels']
        else:
            self.channels = list(data.keys())
        self.update_channels()
        ds, fname = self.setup_filename(runno, name, overwrite)
        self.ds = ds
        data_dir = Path(os.path.dirname(fname))
        if not data_dir.exists():
            data_dir.mkdir(parents=True)
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
                try:
                    data[bs].store(f, group)
                except Exception as e:
                    print(f'ERROR: channel {bs} could not be saved!')
                    print(e)

                if not bs in chsgrp.keys():
                    chsgrp.update({bs:group})

            if self.multi_limits:
                jfs_rois = self.setup_rois_multi_limits(data)
            elif not self.no_jfutils:
                jf_handlers = self.setup_data_handlers(scan_info)
                jfs_rois = self.setup_rois_ju(jf_handlers, data)
            else:
                jfs_rois = self.setup_rois(data)


            if len(jfs_rois.keys()) > 0:
                print('Computing JFs')

            if self.do_slice:
                for jf, jf_rois in jfs_rois.items():
                    rois_comp = []
                    rois_grps = []
                    for rois, roi_datas in jf_rois.items():
                        for roi, roi_data in roi_datas.items():
                            group = '/data/'+jf+'/'+rois+'/'+roi
                            if not jf+'_'+rois+'_'+roi in chsgrp.keys():
                                chsgrp.update({jf+'_'+rois+'_'+roi:group})
                            steps = int(len(roi_data.index)/self.maxshots)
                            if len(roi_data.index) - steps*self.maxshots >0:
                                steps = steps+1
                            print(steps)
                            # roi_data.set_h5_storage(f,group)
                            rois_comp.append([roi_data[step*self.maxshots:(step+1)*self.maxshots] for step in range(steps) ])
                            rois_grps.append(group)


                    #return rois_comp
                    print('Computing JF {}'.format(jf))
                    for step in zip(*rois_comp):
                        for grp, roi in zip(rois_grps, step):
                            roi.set_h5_storage(f,grp)
                        esc.storage.store(step)
            else:
                for jf, jf_rois in jfs_rois.items():
                    rois_comp = []
                    for rois, roi_datas in jf_rois.items():
                        for roi, roi_data in roi_datas.items():
                            group = '/data/'+jf+'/'+rois+'/'+roi
                            if not jf+'_'+rois+'_'+roi in chsgrp.keys():
                                chsgrp.update({jf+'_'+rois+'_'+roi:group})
                            #roi_data.store(f,group)
                            print('did look at indexes',len(roi_data.index))
                            roi_data.set_h5_storage(f,group)
                            rois_comp.append(roi_data)
                    print('Computing JF {}'.format(jf))
                    esc.storage.store(rois_comp)





#class Data_Reduction_zarr():
#
#    def __init__(
#        self,
#        cfg_file,
#    ):
#
#        self.lock = SerializableLock()
#
#        """load config file"""
#        self.cfg_file = cfg_file
#        self.cfg= {}
#        self.update_cfg()
#        """global parameters"""
#        self.dir_save = '/sf/bernina/data/{}/work/analysis/{}/'.format(self.cfg['Exp_config']['pgroup'] ,self.cfg['Exp_config']['save_folder'])
#        self.channels = {}
#        self.ds = {}
#    def update_cfg(self, cfg_file=None):
#        if not cfg_file:
#            cfg_file=self.cfg_file
#        if type(cfg_file) == dict:
#            self.cfg = cfg_file
#        else:
#            with Path(cfg_file).open(mode='r') as f: self.cfg=json.load(f)
#            """convert string True, False and None into booleans"""
#            for key, value in self.cfg['Exp_config'].items():
#                if value == "True":
#                    self.cfg['Exp_config'][key]=True
#                if value == "False":
#                    self.cfg['Exp_config'][key]=False
#                if value == "None":
#                    self.cfg['Exp_config'][key]=None
#            for jf, cfg in self.cfg['JFs'].items():
#                for key, value in cfg.items():
#                    if value == "True":
#                        self.cfg['JFs'][jf][key]=True
#                    if value == "False":
#                        self.cfg['JFs'][jf][key]=False
#                    if value == "None":
#                        self.cfg['JFs'][jf][key]=None
#
#
#
#    def parse_scan(self, runno, exclude_files):
#        pgroup = self.cfg['Exp_config']['pgroup']
#        json_dir = Path(f'/sf/bernina/data/{pgroup}/res/scan_info/')
#        json_file = list(json_dir.glob(f'run{runno:04}*'))[0]
#        print(f'Parsing {json_file}')
#        data = sf.parse_scan(json_file,createEscArrays=True)
#        info, info_path = sf.parse.readScanEcoJson_v01(json_file)
#        return data, info
#
#    def parse_step_from_scan(self, runno, step, exclude_files):
#        pgroup = self.cfg['Exp_config']['pgroup']
#        json_dir = Path(f'/sf/bernina/data/{pgroup}/res/scan_info/')
#        json_file = list(json_dir.glob(f'run{runno:04}*'))[0]
#        print(f'Parsing {json_file}')
#        data = sf.parseScanEco_v01(json_file,createEscArrays=True, memlimit_mD_MB=50,exclude_from_files= exclude_files)
#        info, info_path = sf.readScanEcoJson_v01(json_file)
#        return data, info
#
#    def apply_gain_pede_np(self, image, G, P, pixel_mask=None, mask_value = np.nan):
#        # gain and pedestal correction
#        mask = int('0b' + 14 * '1', 2)
#        mask2 = int('0b' + 2 * '1', 2)
#        gain_mask = np.bitwise_and(np.right_shift(image, 14), mask2)
#        data = np.bitwise_and(image, mask)
#        m1 = gain_mask == 0
#        m2 = gain_mask == 1
#        m3 = gain_mask >= 2
#        if G is not None:
#            g = m1*G[0] + m2*G[1] + m3*G[2]
#        else:
#            g = np.ones(data.shape, dtype=np.float64)
#        if P is not None:
#            p = m1*P[0] + m2*P[1] + m3*P[2]
#        else:
#            p = np.zeros(data.shape, dtype=np.float64)
#        res = np.divide(data - p, g)
#        if pixel_mask is not None:
#		#dv,mv = np.broadcast_arrays(res,pixel_mask) # seems not necessary
#            mv = pixel_mask
#            if isinstance(image,da.Array):
#                mv = da.from_array(mv, chunks=image.chunks[-2:])
#            res[mv!=0] = mask_value
#            res=da.nan_to_num(res,0)
#        return res
#
#    def apply_threshold(self, data, threshold):
#        data[data<threshold] = 0
#        return data
#
#    def apply_energy_limits(self, data, energy_limits):
#        data[data<np.min(energy_limits)] = 0
#        data[data>np.max(energy_limits)] = 0
#        return data
#
#    def convert_to_photons(self, data, energy, delta_e):
#        data[data<energy-delta_e]=0
#        data[np.array([ np.all([m,n]) for m,n in zip(np.remainder(abs(data),energy)>delta_e,np.remainder(abs(data),energy)<energy-delta_e)])]=0
#        data=np.floor_divide(data+delta_e,energy)
#        return data
#
#    def setup_ds(self, fname):
#        ds = dsg.DataStorage(filename = fname)
#        print('Setting up datastorage file')
#        ds['config'] = self.cfg
#        ds['channels'] = {}
#        self.ds = ds
#        ds.save()
#        return ds
#
#    def setup_filename(self, runno, name, overwrite):
#        fname = '{}run_{}_cfg.h5'.format(self.dir_save, runno)
#        fzname = '{}run_{}.zarr'.format(self.dir_save, runno)
#        if name:
#            fname = '{}run_{}_{}_cfg.h5'.format(self.dir_save, runno, name)
#            fzname = '{}run_{}_{}.zarr'.format(self.dir_save, runno, name)
#        if os.path.isfile(fname):
#            print('The file {} already exists'.format(fname))
#            if overwrite == 'question':
#                mes = input('Do you want to remove (y,n) or rename (type new name):')
#                if mes == 'y':
#                    mes = True
#                elif mes == 'n':
#                    mes = False
#            else:
#                mes = overwrite
#                print(mes)
#            if mes:
#                os.remove(fname)
#                ds = self.setup_ds(fname)
#                ds.save()
#            elif not mes:
#                ds = dsg.DataStorage(filename = fname)
#            else:
#                fname = '{}run_{}_{}.h5'.format(self.dir_save, runno, mes)
#                ds = self.setup_ds(fname)
#                ds.save()
#        else:
#            ds = self.setup_ds(fname)
#            ds.save()
#        print(fname)
#        return ds, fname, fzname
#
#
#
#    def setup_rois_nojf_utils(self, data):
#       for jf, jf_cfg in self.cfg['JFs'].items():
#
#        jfs_rois={}
#        for jf, jf_cfg in self.cfg['JFs'].items():
#            pedestal_file = jf_cfg['pedestal_file']
#            gain_file = jf_cfg['gain_file']
#            with h5py.File(f'/sf/jungfrau/config/gainMaps/{jf_cfg["id"]}/gains.h5','r') as fh:
#                gain = fh['gains'][:]
#            with h5py.File(pedestal_file,'r') as fh:
#                pedestal = fh['gains'][:]
#            try:
#                jf_data = data[jf_cfg['id']]
#                dummy = jf_data.index
#                jf_data = jf_data.map_index_blocks(self.apply_gain_pede_np,gain,pedestal,dtype=np.float64)
#                threshold = jf_cfg['threshold']
#                if not threshold == None:
#                    jf_data = jf_data.map_index_blocks(self.apply_threshold, threshold=threshold,dtype=np.float64)
#                if 'photon_conversion' in jf_cfg:
#                    print('Converting to photons')
#                    delta_e = jf_cfg['photon_conversion']['delta_e']
#                    energy = jf_cfg['photon_conversion']['energy']
#                    jf_data = jf_data.map_index_blocks(self.convert_to_photons, energy=energy, delta_e=delta_e,dtype=np.float64)
#                energy_limits = jf_cfg['energy_limits']
#                if not energy_limits == None:
#                    print('Applying the super cool new energy limits!')
#                    jf_data = jf_data.map_index_blocks(self.apply_energy_limits, energy_limits=energy_limits,dtype=np.float64)
#                jfs_rois[jf]={'rois':{}, 'rois_img':{}}
#                for roi, roi_rng in jf_cfg['rois'].items():
#                    r = roi_rng
#                    jfs_rois[jf]['rois'][roi] = jf_data[:,r[0]:r[1],r[2]:r[3]].nansum(axis=(1,2))
#                for roi, roi_rng in jf_cfg['rois_img'].items():
#                    r=roi_rng
#                    jfs_rois[jf]['rois_img'][roi] = jf_data[:,r[0]:r[1],r[2]:r[3]]
#            except KeyError:
#                print('JF {} with id ({}) not in parsed data'.format(jf,jf_cfg['id']))
#
#        return jfs_rois
#
#
#    def showdata(self, runno, step,exclude_files, jf=None, maxshots = None, clim=(0,20), hist_lineout = False):
#        if not jf:
#            jf = 'JFscatt'
#        jf_cfg = self.cfg['JFs'][jf]
#        pgroup = self.cfg['Exp_config']['pgroup']
#        pedestal_file = jf_cfg['pedestal_file']
#
#        json_dir = Path(f'/sf/bernina/data/{pgroup}/res/scan_info/')
#        print(json_dir)
#        json_file = list(json_dir.glob(f'run{runno:04}*'))[0]
#        data = {}
#        p = Path(json_file)
#        with p.open(mode="r") as f:
#            s = json.load(f)
#            f = [f for f in s['scan_files'][step] if jf_cfg['id'] in f][0]
#        with ju.File(file_path= f, pedestal_file=pedestal_file, geometry=False, gap_pixels=False) as juf:
#            if maxshots:
#                imgs_ju = juf[:maxshots]
#            else:
#                imgs_ju = juf[:]
#            threshold = jf_cfg['threshold']
#            if not threshold == None:
#                imgs_ju[imgs_ju<threshold]=0
#            energy_limits = jf_cfg['energy_limits']
#            if not energy_limits == None:
#                imgs_ju[imgs_ju<np.min(energy_limits)]=0
#                imgs_ju[imgs_ju>np.max(energy_limits)]=0
#            nshots = len(imgs_ju)
#            img = np.nansum(imgs_ju,axis=0)
#            def hist_only_y(*args,**kwargs):
#                y, bins = np.histogram(*args,**kwargs)
#                return y
#            bins = np.linspace(-5,50,400)
#            if hist_lineout:
#                hists_vertical = np.apply_along_axis(hist_only_y, 1,np.hstack(imgs_ju), bins=bins)
#                data['hists_vertical']=hists_vertical
#            hist_av = np.histogram(imgs_ju[:100], bins=bins)
#            data['hist_av']=hist_av
#        fig = plt.figure(num='Run_{}, Step_{}. Nbr_shots = {}'.format(runno,step, nshots), figsize=(9,9))
#        ax0= fig.add_subplot(1,2,1)
#        ax0.imshow(img,clim=clim,origin='lower')
#        ax1=[]
#
#        rois = jf_cfg['rois']
#        rois_img = jf_cfg['rois_img']
#
#        print(rois)
#        print(rois_img)
#        for i in range(1,len(rois)+len(rois_img)+1):
#            ax1.append(fig.add_subplot(len(rois)+len(rois_img),2,2*i))
#
#        for i,(r,r_rng)  in enumerate(rois.items()):
#            ax1[i].imshow(self.mkroi(img, r_rng),clim=clim)
#            ax1[i].set_title('{}'.format(r))
#            rr =r_rng
#            ax0.add_patch(Rectangle((rr[2], rr[0]), abs(rr[2]-rr[3]),abs(rr[1]-rr[0]),fill=None,
#                          alpha=1,linewidth=2, edgecolor='r'))
#
#        for i,(r,r_rng)  in enumerate(rois_img.items()):
#            i +=len(rois.keys())
#            ax1[i].imshow(self.mkroi(img, r_rng),clim=clim)
#            ax1[i].set_title('Img {}'.format(r))
#            rr =r_rng
#            ax0.add_patch(Rectangle((rr[2], rr[0]), abs(rr[2]-rr[3]),abs(rr[1]-rr[0]),fill=None,
#                          alpha=1,linewidth=2, edgecolor='g'))
#        plt.tight_layout()
#        hist = plt.figure('histogram', figsize=(9,3))
#        plt.plot(hist_av[1][:-1],hist_av[0])
#        plt.xlabel('energy (keV)')
#        plt.ylabel('px/bin')
#        plt.yscale('log')
#        plt.ylim(hist_av[0][-1],np.max(hist_av[0]))
#        plt.tight_layout()
#        data['img']=img
#        data['fig']=fig
#        data['hist_bins']=bins
#        return data
#
#
#    def mkroi(self, dat, r):
#        return dat[r[0]:r[1],r[2]:r[3]]
#
#
#    def update_channels(self):
#        jf_ids = [self.cfg['JFs'][jf]['id'] for jf in self.cfg['JFs'].keys()]
#        bs_ids =  [self.cfg['BSs'][bs]['id'] for bs in self.cfg['BSs'].keys()]
#        for jf_id in jf_ids:
#            try:
#                self.channels.remove(jf_id)
#            except:
#                print('{} not in parsed data'.format(jf_id))
#
#        self.cfg['other']=self.channels
#
#    def data_reduction(self, runno, exclude_files=['IMAGE'],name=False, overwrite='question'):
#        data, scan_info = self.parse_scan(runno,exclude_files)
#        poss = np.squeeze(scan_info['scan_values'])
#
#        if 'BS_channels' in self.cfg.keys():
#            self.channels = self.cfg['BS_channels']
#        else:
#            self.channels = list(data.keys())
#        self.update_channels()
#
#        ds, fname, fzname = self.setup_filename(runno, name, overwrite)
#        self.ds = ds
#
#        with zarr.open(fzname+'.zarr', 'w') as f:
#            chsgrp = f.require_group('channels')
#
#            for bs in self.channels:
#                if 'processing' in bs:
#                    continue
#                if 'FPICTURE' in bs:
#                    continue
#                print('Computing {}'.format(bs))
#                group = '/data/'+bs
#                dummy = data[bs].index
#                try:
#                    data[bs].store(f, group, lock=self.lock)
#                except Exception as e:
#                    print(f'ERROR: channel {bs} could not be saved!')
#                    print(e)
#
#                if not bs in chsgrp.keys():
#                    chsgrp.update({bs:group})
#
#
#            jfs_rois = self.setup_rois_nojf_utils(data)
#            if len(jfs_rois.keys()) > 0:
#                print('Computing JFs')
#
#            for jf, jf_rois in jfs_rois.items():
#                rois_comp = []
#                for rois, roi_datas in jf_rois.items():
#                    for roi, roi_data in roi_datas.items():
#                        group = '/data/'+jf+'/'+rois+'/'+roi
#                        if not jf+'_'+rois+'_'+roi in chsgrp.keys():
#                            chsgrp.update({jf+'_'+rois+'_'+roi:group})
#                        #roi_data.store(f,group)
#                        print('did look at indexes',len(roi_data.index))
#                        roi_data.set_h5_storage(f,group)
#                        rois_comp.append(roi_data)
#                print('Computing JF {}'.format(jf))
#                esc.storage.store(rois_comp, lock=self.lock)
