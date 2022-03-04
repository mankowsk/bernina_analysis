from ..utilities.utilities import *
import jungfrau_utils as ju
import matplotlib.pyplot as plt
import numpy as np
import dask.array as da
import json
import os
import logging
from pathlib import Path
import sys

sys.path.insert(0, "/sf/bernina/config/src/python/escape_dev/")
import escape.parse.swissfel as sf
import h5py
import escape as esc
import datastorage as dsg
from ipywidgets import interact, interactive, FloatSlider
from matplotlib.patches import Rectangle
from IPython.display import display
from scipy.signal import savgol_filter
from scipy import signal as Signal
from lmfit import Model, models, Parameters
from scipy.special import erf
from scipy.optimize import curve_fit
import scipy
from pathlib import Path
#import zarr


def loaddata(f, chunkreduction_hack=True):
    f = h5py.File(f, "r")
    data = {
        ch: esc.Array.load_from_h5(f, f["channels"][ch][()])
        for ch in f["channels"].keys()
    }
    if chunkreduction_hack:
        names = ["SAR-CVME-TIFALL5:EvtSet", "", ""]
        for name in names:
            try:
                refdataset = data[name]
                print("found")
                break
            except:
                continue
        for name, arr in data.items():
            if len(arr.scan) == 1:
                narr = (
                    refdataset.ones().__getitem__(
                        tuple([slice(None)] + [None] * (arr.ndim - 1))
                    )
                    * arr
                )
                dummy = narr.index
                data[name] = narr
                print(f"Fixed escape array {name} for missing scan information.")

    return data


#def loaddata_zarr(f):
#    f = zarr.load(f, "a")
#    data = {
#        ch: esc.Array.load_from_h5(f, f["channels"][ch][()])
#        for ch in f["channels"].keys()
#    }
#    return data


### To do: class run with functions to average, save, which can be summed up with another run to create a dictionary of runs and apply the timetool correction, filters and so on to multiple runs.


def analyse_filter(
    runno,
    data = None,
    filters={"i0_sum": [0, 700]},
    sig=["jf_pk"],
    i0=["jf_fl"],
    plot_hist=False,
    plot=True,
    save=False,
    noimg=True,
    name="",
):
    datas, evts, filt, filt_ea, filt_ratio = apply_filter(
        runno, data=data, filters=filters, sig=sig, plot_hist=plot_hist, noimg=noimg, name=name
    )

    xname = list(evts.scan.parameter)[1]
    xrbname = list(evts.scan.parameter)[2]
    xset = evts.scan.parameter[xname]["values"]
    xrb = evts.scan.parameter[xrbname]["values"]

    # filter data
    datas_filt = {
        key: {ll: value[ll][filt[ll]] for ll in ["on", "off"]}
        for key, value in datas.items()
    }

    # create x axis from remaining steps
    first_jf = [key for key in datas_filt.keys() if "jf" in key][0]

    xset_filt = np.array(
        [
            step
            for step in datas_filt[first_jf]["off"].scan.parameter[xname]["values"]
            if step in datas_filt[first_jf]["on"].scan.parameter[xname]["values"]
        ]
    )
    xrb_filt = np.array(
        [
            step
            for step in datas_filt[first_jf]["off"].scan.parameter[xrbname]["values"]
            if step in datas_filt[first_jf]["on"].scan.parameter[xrbname]["values"]
        ]
    )

    exclude = ["CH", "tt"]
    if noimg:
        exclude = exclude + ["img"]
    print(exclude)
    # average step data
    datas_av = {
        key: {
            ll: np.array([np.nansum(step.data, axis=0) for step in value[ll].scan])
            for ll in ["on", "off"]
        }
        for key, value in datas.items()
        if not np.any([ch in key for ch in exclude])
    }
    datas_filt_av = {
        key: {
            ll: np.array(
                [
                    np.nansum(step.data, axis=0)
                    for step, step_pos in zip(
                        value[ll].scan, value[ll].scan.parameter[xname]["values"]
                    )
                    if step_pos in xset_filt
                ]
            )
            for ll in ["on", "off"]
        }
        for key, value in datas_filt.items()
        if not np.any([ch in key for ch in exclude])
    }

    if plot:
        i0ch = i0[0]
        jfs = {key: value for key, value in datas_av.items() if "jf" in key if not i0ch in key if not 'img' in key}
        jfs_filt = {key: value for key, value in datas_filt_av.items() if "jf" in key if not i0ch in key}

        fig, ax = plt.subplots(len(jfs.keys()), 2, figsize=(10, 4 * len(jfs.keys())), num=runno)

        for n, ((jf, jfd), (jf_filt, jfd_filt)) in enumerate(
            zip(jfs.items(), jfs_filt.items())
        ):
            ax[n][0].plot(
                xset,
                (jfd["on"] / datas_av[i0ch]["on"])
                / (jfd["off"] / datas_av[i0ch]["off"]),
                "g",
                label=f"{jf} ratio",
            )
            ax[n][0].plot(
                xset_filt,
                (jfd_filt["on"] / datas_filt_av[i0ch]["on"])
                / (jfd_filt["off"] / datas_filt_av[i0ch]["off"]),
                "b",
                label=f"{jf_filt} filtered ratio",
            )
            ax[n][0].set_xlabel(f"{xname}")
            ax[n][0].set_ylabel(f"on/off i0 normalized")
            ax[n][1].set_ylabel(f"i/ {i0ch}")
            ax[n][0].set_title(f"Roi {jf}")

            ax[n][1].set_title(f"Roi {jf}")

            ax[n][1].plot(
                xset, (jfd["on"] / datas_av[i0ch]["on"]), "g", label=f"{jf} on"
            )
            ax[n][1].plot(
                xset, (jfd["off"] / datas_av[i0ch]["off"]), "orange", label=f"{jf} off"
            )

            ax[n][1].plot(
                xset_filt,
                (jfd_filt["on"] / datas_filt_av[i0ch]["on"]),
                "blue",
                label=f"{jf_filt} filt on",
            )
            ax[n][1].plot(
                xset_filt,
                (jfd_filt["off"] / datas_filt_av[i0ch]["off"]),
                "red",
                label=f"{jf_filt} filt off",
            )
            ax[n][0].legend()
            ax[n][1].legend()
        fig.suptitle(f'Run {runno}')
        fig.tight_layout()

    datas_av.update(
        {
            "xset": xset,
            "xrb": xrb,
            "xname": xname,
        }
    )
    datas_filt_av.update(
        {
            "xset": xset_filt,
            "xrb": xrb_filt,
            "xname": xname,
        }
    )
    full_data = {
        "data_av": datas_av,
        "data_filt_av": datas_filt_av,
        "used_shots": filt_ratio,
    }

    if save:
        dsg.save(f"text_data/run_{runno}_filt_dev.h5", full_data)
        Path(f"text_data/run_{runno}_filt_dev.h5").chmod(0o775)

    return full_data


def apply_filter(
    runno,
    data = None,
    filters={"i0_sum": [0, 700], "i0_x": "auto", "i0_y": "auto"},
    sig=["jf_JFXRD_rois_pk"],
    plot_hist=False,
    noimg=False,
    tt=False,
    pid_offset=0,
    name="",
    dir_name="small_data",
):
    if data is None:
        data = loaddata(f"{dir_name}/run_{runno}{name}.h5")
    if noimg:
        # jfs = {f'jf_{key[13:]}':data[key] for key in data.keys() if 'JF' in key and 'img' not in key}
        jfs = {
            f"jf_{key}": data[key]
            for key in data.keys()
            if "JF" in key and "img" not in key
        }
    else:
        # jfs = {f'jf_{key[13:]}':data[key] for key in data.keys() if 'JF' in key}
        jfs = {f"jf_{key}": data[key] for key in data.keys() if "JF" in key}
    evts = data["SAR-CVME-TIFALL5:EvtSet"].compute()
    i0s = {
        f"i0_{key[18:21]}": correct_ioxos_shape(data[key])
        for key in data.keys()
        if "SLAAR21-LSCP1-FNS:" in key
    }
    try:
        i0s.update(
            {
                f"i0_138_CH{n}": data[f'SAROP21-CVME-PBPS2:Lnk9Ch{n}-DATA-SUM'] for n in range(1,5)
            }

        )
        i0s["i0_sum_138"] = (
            i0s["i0_138_CH3"] - i0s["i0_138_CH4"] + i0s["i0_138_CH3"] + i0s["i0_138_CH4"]
        )
        i0s["i0_x_138"] = (
            i0s["i0_138_CH3"]/np.median((i0s["i0_138_CH3"].data).compute()) - i0s["i0_138_CH4"]/np.median((i0s["i0_138_CH4"].data).compute())
        )
        i0s["i0_y_138"] = (
            i0s["i0_138_CH1"]/np.median((i0s["i0_138_CH1"].data).compute()) - i0s["i0_138_CH2"]/np.median((i0s["i0_138_CH2"].data).compute())
        )

    except:
        print('PBPS 138 not in data')
    if tt:
        tt_chs = {
            "tt_sig": mod_pid(
                data["SARES20-CAMS142-M5.roi_signal_x_profile"], pid_offset
            )
        }
    else:
        tt_chs = {}
    datas = {**jfs, **i0s, **tt_chs}

    # add the sum of the i0 diodes
    datas["i0_sum"] = (
        datas["i0_CH4"] + datas["i0_CH5"] + datas["i0_CH6"] + datas["i0_CH7"]
    )


    datas.update(
        calc_i0_pos_ea(
            *esc.match_arrays(
                datas["i0_CH4"], datas["i0_CH5"], datas["i0_CH6"], datas["i0_CH7"]
            )
        )
    )

    # compute arrays used for filter
    datas.update(
        {
            ch: val.compute()
            for ch, val in datas.items()
            if not np.any([tt_ch in ch for tt_ch in tt_chs.keys()])
        }
    )

    # match arrays, compute, sort in on and off
    datas = {
        ch: val
        for ch, val in zip(
            datas.keys(), on_off(esc.match_arrays(*[ea for ea in datas.values()]), evts)
        )
    }

    # create filter array
    filt = {}
    filt_ea = {}
    filt_ratio = {}
    for las in ["on", "off"]:
        mask = []
        for filt_ch, filt_th in filters.items():
            if filt_th == "auto":
                filt_std = np.std(
                    datas[filt_ch][las][
                        (datas[filt_ch][las].data - np.median(datas[filt_ch][las].data))
                        < 2 * np.std(datas[filt_ch][las].data)
                    ]
                )
                filt_med = np.median(datas[filt_ch][las].data)
                bins = np.linspace(-4 * filt_std, 4 * filt_std, 200)
                hist, bins = np.histogram(datas[filt_ch][las].data - filt_med, bins)
                fwhm = 2 * abs(bins[np.argmin(abs(hist - 0.5 * np.max(hist)))])
                filt_th = [filt_med - 1.5 * fwhm, filt_med + 1.5 * fwhm]
                filters[filt_ch] = filt_th
                print(
                    f"Auto set {filt_ch} threshold to [{filt_th[0]:.3},{filt_th[1]:.3}]"
                )
            mask.append(
                (datas[filt_ch][las].data > filt_th[0])
                * (datas[filt_ch][las].data < filt_th[1])
            )
        mm = mask[0]
        for m in mask:
            mm = mm & m
        filt[las] = mm
        filt_ea[las] = esc.Array(index=datas["i0_sum"][las].index, data=mm)
        if len(mm) == 0:
            print(f'No {las} shots in the data')
        else:
            filt_ratio[las] = sum(mm) / len(mm) * 100
            print(f"After filtering {sum(mm)/len(mm)*100}% of the {las} pulses remaining")

    # plot histograms of the filters
    if plot_hist:
        print(f'Keys in data: {datas.keys()}')
        plt.ion()
        naxs = len(filters.keys())
        if naxs < 2:
            naxs = 2
        fig, ax = plt.subplots(
            len(filters.keys()), 2, figsize=(8, 2.5 * (len(filters.keys())))
        )
        for n, (k, th) in enumerate(filters.items()):

            d_std = np.std(
                datas[k]["off"][
                    (datas[k]["off"].data - np.median(datas[k]["off"].data))
                    < 2 * np.std(datas[k]["off"].data)
                ]
            )
            d_med = np.median(datas[k]["off"].data)
            bins = np.linspace(d_med - 4 * d_std, d_med + 4 * d_std, 200)
            print(n)
            ax[n][0].hist(datas[k]["off"].data, bins=bins)
            ax[n][0].set_title(f"Run {runno} {k} histogram")
            ax[n][0].axvline(th[0], linewidth=1, color="red")
            ax[n][0].axvline(th[1], linewidth=1, color="red")
            ax[n][1].plot(
                datas[k]["off"].data[:2000],
                datas[sig[0]]["off"].data[:2000],
                ".",
                markersize=1,
            )
            ax[n][1].plot(
                (datas[k]["off"][:2000])[mask[n][:2000]].data,
                (datas[sig[0]]["off"][:2000])[mask[n][:2000]].data,
                ".",
                label="filter",
                markersize=1,
            )
            ax[n][1].plot(
                (datas[k]["off"][:2000])[filt["off"][:2000]].data,
                (datas[sig[0]]["off"][:2000])[filt["off"][:2000]].data,
                ".",
                label="all_filter",
                markersize=1,
            )
            # ax[n][0].set_xlim([bins[0],bins[-1]])
            ax[n][1].set_xlabel(k)
            ax[n][1].set_ylabel(sig[0])
            ax[n][1].legend()
        fig.tight_layout()
        plt.show()
        print("Use these filters (y) or break evaluation?")
        x = input()
        if x is not "y":
            raise Exception("User not happy with filters")
    return datas, evts, filt, filt_ea, filt_ratio


def analyse_filter_timetool(
    runno,
    save=False,
    calib=1.9e-3,
    bin_size=None,
    overwrite_tt=False,
    sub_median=True,
    pid_offset=0,
    filters={"i0_sum": [0, 700]},
    sig=["jf_pk"],
    i0=["jf_fl"],
    plot_hist=False,
    noimg=True,
    tt_offset=0,
    tt_amp_min=0,
    edge_width=100,
    px_smooth=10,
    corr_by_max=False,
    dir_name="small_data",
    name=''
):
    tt_filename = f"{dir_name}/run_{runno}_jitter.h5"
    datas, evts, filt, filt_ea, filt_ratio = apply_filter(
        runno,
        filters=filters,
        sig=sig,
        plot_hist=plot_hist,
        noimg=noimg,
        tt=True,
        pid_offset=pid_offset,
        name=name
    )
    data = loaddata(f"{dir_name}/run_{runno}.h5")

    # filter data
    datas_filt = {
        key: {ll: value[ll][filt[ll]] for ll in ["on", "off"]}
        for key, value in datas.items()
    }
    xname = list(evts.scan.parameter)[0]
    xrbname = list(evts.scan.parameter)[1]
    xset = np.array(evts.scan.parameter[xname]["values"]) * 1e12
    xrb = np.array(evts.scan.parameter[xrbname]["values"]) * 1e12
    pos_all = np.concatenate(
        [
            (np.zeros((step_length,)) + step_pos)
            for step_length, step_pos in zip(
                datas_filt["tt_sig"]["on"].scan.step_lengths, xset
            )
        ]
    ).T
    first_jf = [key for key in datas_filt.keys() if "jf" in key][0]

    xset_filt = np.array(
        [
            step
            for step in datas_filt[first_jf]["off"].scan.parameter[xname]["values"]
            if step in datas_filt[first_jf]["on"].scan.parameter[xname]["values"]
        ]
    )
    xrb_filt = np.array(
        [
            step
            for step in datas_filt[first_jf]["off"].scan.parameter[xrbname]["values"]
            if step in datas_filt[first_jf]["on"].scan.parameter[xrbname]["values"]
        ]
    )

    exclude = ["CH", "tt"]
    if noimg:
        exclude = exclude + ["img"]
    print("calculating average step data")
    datas_av = {
        key: {
            ll: np.array([np.nansum(step.data, axis=0) for step in value[ll].scan])
            for ll in ["on", "off"]
        }
        for key, value in datas.items()
        if not np.any([ch in key for ch in exclude])
    }
    print("calculating average filtered step data")
    datas_filt_av = {
        key: {
            ll: np.array(
                [
                    np.nansum(step.data, axis=0)
                    for step, step_pos in zip(
                        value[ll].scan, value[ll].scan.parameter[xname]["values"]
                    )
                    if step_pos in xset_filt
                ]
            )
            for ll in ["on", "off"]
        }
        for key, value in datas_filt.items()
        if not np.any([ch in key for ch in exclude])
    }

    xset_filt = xset_filt * 1e12
    xrb_filt = xrb_filt * 1e12

    # time tool correction
    width = edge_width
    erf_ratio = erf_edge(datas["tt_sig"]["on"].shape[1], width)
    roi = [
        int((datas["tt_sig"]["on"].shape[1]) / 2 - width),
        int((datas["tt_sig"]["on"].shape[1]) / 2 + width),
    ]

    if not overwrite_tt and os.path.exists(tt_filename):
        with h5py.File(tt_filename, "r") as f_tt:
            edge_pos_ea = esc.Array.load_from_h5(f_tt, "edge_pos").compute()
            edge_amp_ea = esc.Array.load_from_h5(f_tt, "edge_amp").compute()
    else:
        edge_pos_ea, edge_amp_ea = analyse_edge_correlation(
            datas["tt_sig"], ratio_av=erf_ratio, roi=roi, px_smooth=px_smooth
        )
        with h5py.File(Path(tt_filename).expanduser(), "w") as f_tt:
            edge_pos_ea.store(f_tt, "edge_pos")
            edge_amp_ea.store(f_tt, "edge_amp")
            edge_pos_ea = edge_pos_ea.compute()
            edge_amp_ea = edge_amp_ea.compute()

    print("edge_pos_ea before matching", edge_pos_ea.shape)

    filt_ea["on"], edge_pos_ea, edge_amp_ea = esc.match_arrays(
        filt_ea["on"], edge_pos_ea, edge_amp_ea
    )
    print("edge_pos_ea after matching", edge_pos_ea.shape)

    edge_pos_ea = edge_pos_ea[filt_ea["on"].data]
    edge_amp_ea = edge_amp_ea[filt_ea["on"].data]
    tt_filt = edge_amp_ea.data > tt_amp_min
    print(
        f"Rejecting {(len(tt_filt)-sum(tt_filt))/len(tt_filt)*100}% of shots because of bad tt results"
    )

    if sub_median:
        print(f"subtracting median of {np.median(edge_pos_ea.data)} from edge_pos")
        edge_pos_ps = edge_pos_ea.data - np.median(edge_pos_ea.data)
    else:
        print(f"subtracting fixed value of {tt_offset} from edge_pos")
        edge_pos_ps = edge_pos_ea.data - tt_offset
        print(
            f"difference between median and fixed value: {np.median(edge_pos_ea.data)-tt_offset}"
        )

    if not bin_size:
        bin_size = bin_size = abs(xset[1] - xset[0])
    edge_pos_ps = edge_pos_ps * calib
    if corr_by_max:
        on = datas_av["jf_pk"]["on"] / datas_av["jf_fl"]["on"]
        off = datas_av["jf_pk"]["off"] / datas_av["jf_fl"]["off"]
        idx = np.nanargmax(abs(on / off - 1))
        pos_all = pos_all - datas_av["xset"][idx]
        print(f'max of jf_pk signal at idx {idx} at time {datas_av["xset"][idx]}')

    pos_corr = pos_all - edge_pos_ps
    tt_bins = (
        np.arange(np.nanmin(xset), np.nanmax(xset) - bin_size / 2, bin_size)
        + bin_size / 2
    )
    tt_pos = np.arange(np.nanmin(xset), np.nanmax(xset) + bin_size / 2, bin_size)

    ### bins need to be shorter by 1 and shifted by half a
    ### bin_size compared to the real time points as np.digitize
    ### uses < bins[0] as '0' and > bins [-1] as last value
    on_pid = datas_filt["tt_sig"]["on"].index
    off_pid = datas_filt["tt_sig"]["off"].index
    off_idx = np.digitize(off_pid, on_pid[:-1] - 0.5)
    assined_bin_on = np.digitize(pos_corr, tt_bins)
    assigned_bin = {"on": assined_bin_on, "off": assined_bin_on[off_idx]}
    assigned_bin["on"] = (assigned_bin["on"] + 1) * tt_filt - 1
    # print(f'Rejecting {(len(assigned_bin["on"])-len(assigned_bin["on"][assigned_bin["on"]==-1]))/len(assigned_bin["on"])*100}% of shots because of bad tt results')

    print("calculating tt binned average step data")
    datas_filt_av_tt = {
        key: {
            las: np.array(
                [
                    np.nansum(dat[las].data[(to_bin == i)], axis=0)
                    for i in range(len(tt_pos))
                ]
            )
            for las, to_bin in assigned_bin.items()
        }
        for key, dat in datas_filt.items()
        if not np.any([ch in key for ch in exclude])
    }

    datas_filt_av_tt.update(
        {
            "xset": tt_pos,
            "tt_poss": edge_pos_ea.data,
            "tt_amps": edge_amp_ea.data,
            "i0_sum_ons": datas_filt["i0_sum"]["on"].data,
        }
    )

    datas_av.update(
        {
            "xset": xset,
            "xrb": xrb,
            "xname": xname,
        }
    )
    datas_filt_av.update(
        {
            "xset": xset_filt,
            "xrb": xrb_filt,
            "xname": xname,
        }
    )
    full_data = {
        "data_av": datas_av,
        "data_filt_av": datas_filt_av,
        "data_filt_av_tt": datas_filt_av_tt,
        "used_shots": filt_ratio,
    }
    if save:
        print("saving")
        if type(save) is str:
            filename = save
        else:
            filename = f"text_data/run_{runno}_filt_tt.h5"
        dsg.save(filename, full_data)
        Path(filename).chmod(0o775)

    return full_data


def correlation(runno, dir_name="small_data"):
    data = loaddata(f"{dir_name}/run_{runno}.h5")
    Evts = data["SAR-CVME-TIFALL5:EvtSet"].compute()
    rd = dict()

    for k in list(data.keys()):
        if "JFscatt_rois_" in k:
            rd[k] = []

    i0 = data["SLAAR21-LTIM01-EVR0:CALCI"].compute()
    i0, Evts = esc.match_arrays(i0, Evts)
    for arg in rd:
        rd[arg], Evts = esc.match_arrays(data[arg].compute(), Evts)

    for arg in rd:
        rd[arg] = on_off([rd[arg]], Evts)[0]

    i0 = on_off([i0], Evts)
    rd.update({"i0": i0})

    return rd


def phiScan(runno, save=True, plot=False, dir_name="small_data"):
    data = loaddata(f"{dir_name}/run_{runno}.h5")
    Evts = data["SAR-CVME-TIFALL5:EvtSet"].compute()
    bg = data["JFscatt_rois_bg"]
    pk = data["JFscatt_rois_pk"]
    fl = data["JFscatt_rois_fl"]
    i0 = data["SLAAR21-LTIM01-EVR0:CALCI"].compute()

    motor_name = list(pk.scan.parameter.keys())[1]
    pos = np.asarray(pk.scan.parameter[motor_name]["values"])
    pk, bg, fl, Evts = esc.match_arrays(pk, bg, fl, Evts)
    i0, pk, bg, fl = on_off([i0, pk, bg, fl], Evts)

    i0_av = dict(on=[], off=[])
    pk_av = dict(on=[], off=[])
    fl_av = dict(on=[], off=[])
    bg_av = dict(on=[], off=[])

    for las in ["on", "off"]:
        i0_av[las] = np.array([np.nanmean(ei.data) for ei in i0[las].scan])
        fl_av[las] = np.array([np.nanmean(ei.data) for ei in fl[las].scan])
        pk_av[las] = np.array([np.nanmean(ei.data) for ei in pk[las].scan])
        bg_av[las] = np.array([np.nanmean(ei.data) for ei in bg[las].scan])

    ra_pk = pk_av["off"]
    ra_fl = fl_av["off"]
    ra_bg = bg_av["off"]
    ra_i0 = i0_av["off"]

    pk_on = (pk_av["on"] - bg_av["on"]) / fl_av["on"]
    pk_off = (pk_av["off"] - bg_av["off"]) / fl_av["off"]
    diff = pk_on / pk_off

    if save:
        np.save(f"text_data/run_{runno}.npy", [pos, pk_on, pk_off, diff])
        Path(f"text_data/run_{runno}.npy").chmod(0o775)

    if plot:
        fig, ax = plt.subplots(1, 2, num="Phi-Scan", figsize=(9, 4))
        fig.suptitle(f"Run {runno}")
        ax[0].plot(pos, (pk_av["on"] - bg_av["on"]) / fl_av["on"], label="on")
        ax[0].plot(pos, (pk_av["off"] - bg_av["off"]) / fl_av["off"], label="off")
        ax[0].set_ylabel("(pk- bg)/fl")
        ax[0].legend()
        ax[1].set_ylabel("On/Off")
        ax[1].plot(
            pos,
            (pk_av["on"] - bg_av["on"])
            / fl_av["on"]
            / ((pk_av["off"] - bg_av["off"]) / fl_av["off"]),
        )
        ax[1].set_xlabel("Phi")
        # fig.tight_layout()
    return pos, pk_on, pk_off, diff
    # return pos,ra_i0, ra_pk, ra_bg,ra_fl,pk_av,fl_av


def phiScanI0pbps(
    runno, save=True, plot=False, what="diff", corr_by="fl", dir_name="small_data"
):
    data = loaddata(f"{dir_name}/run_{runno}.h5")
    Evts = data["SAR-CVME-TIFALL5:EvtSet"].compute()
    bg = data["JFscatt_rois_bg"]
    pk = data["JFscatt_rois_pk"]
    fl = data["JFscatt_rois_fl"]
    i0 = data["SLAAR21-LTIM01-EVR0:CALCI"].compute()

    motor_name = list(pk.scan.parameter.keys())[1]
    pos = np.asarray(pk.scan.parameter[motor_name]["values"])
    i0, bg, fl, Evts = esc.match_arrays(i0, bg, fl, Evts)
    i0, pk, bg, fl = on_off([i0, pk, bg, fl], Evts)

    i0_av = dict(on=[], off=[])
    pk_av = dict(on=[], off=[])
    fl_av = dict(on=[], off=[])
    bg_av = dict(on=[], off=[])

    for las in ["on", "off"]:
        i0_av[las] = np.array([np.nanmean(ei.data) for ei in i0[las].scan])
        fl_av[las] = np.array([np.nanmean(ei.data) for ei in fl[las].scan])
        pk_av[las] = np.array([np.nanmean(ei.data) for ei in pk[las].scan])
        bg_av[las] = np.array([np.nanmean(ei.data) for ei in bg[las].scan])

    ra_pk = pk_av["off"]
    ra_fl = fl_av["off"]
    ra_bg = bg_av["off"]
    ra_i0 = i0_av["off"]

    if corr_by == "i0":
        pk_on = (pk_av["on"]) / i0_av["on"]
        pk_off = (pk_av["off"]) / i0_av["off"]

    elif corr_by == "fl":
        pk_on = (pk_av["on"]) / fl_av["on"]
        pk_off = (pk_av["off"]) / fl_av["off"]

    if what == "diff":
        onoff = pk_on - pk_off
    elif what == "ratio":
        onoff = pk_on / pk_off

    elif what == "on":
        onoff = pk_on
        # pk_off = np.zeros_like(pk_on)

    elif what == "off":
        onoff = pk_off
        # pk_on = np.zeros_like(pk_off)

    if save:
        np.save(f"text_data/run_{runno}.npy", [pos, pk_on, pk_off, onoff])
        Path(f"text_data/run_{runno}.npy").chmod(0o775)

    if plot:
        fig, ax = plt.subplots(1, 2, num=f"Phi-Scan {runno}", figsize=(9, 4))
        fig.suptitle(f"Run {runno}")
        ax[0].plot(pos, pk_on, label="on")
        ax[0].plot(pos, pk_off, label="off")
        ax[0].set_ylabel(f"(pk- bg)/{corr_by}")
        ax[0].legend()
        ax[1].set_ylabel(what)
        ax[1].plot(pos, onoff)
        ax[1].set_xlabel("Phi")
        fig.tight_layout()
    return pos, pk_on, pk_off, onoff


def do_FFT_Time_Freq(t, tr, lm=None, lx=None, a0=100, sig=0.2, plot=True, clm=1):
    td_fft = np.zeros((len(t)))
    for i in range(len(t)):
        spec = tr * gaussian(t, a0=a0, t0=t[i], sig=sig, bg=0)
        # print(spec)
        amp_fft = do_fft(t, spec, lm, lx, plot=False)[1]
        # print(amp_fft)
        td_fft = np.dstack((td_fft, amp_fft))
    freq = do_fft(t, spec, lm, lx, plot=False)[0]
    td_fft = td_fft

    x = t[0:900]
    y = freq[50:200]
    z = td_fft[0][50:200, 0:900]
    #    plt.pcolor(t[10:600],freq[4:25],td_fft[0][4:25,10:600],vmin=-5,vmax=5)
    if plot is True:
        plt.figure(num=3)
        plt.imshow(
            z,
            cmap="jet",
            extent=[x.min(), x.max(), y.min(), y.max()],
            interpolation="spline36",
            origin="lower",
            clim=(z.min(), clm),
            aspect="auto",
        )
        plt.ylabel("Frequency / THz")
        plt.xlabel("dt / ps")

        # plt.axis([-10,-4,1,10])
        plt.tight_layout()
    return x, y, z


def gaussian(t, a0, t0, sig, bg):
    return bg + (
        a0 / np.sqrt(2 * np.pi) / sig * np.exp(-((t - t0) ** 2) / 2 / sig ** 2)
    )


def plot2D(x, y, C, *args, **kwargs):
    def bin_array(arr):
        return np.hstack([arr - np.diff(arr)[0] / 2, arr[-1] + np.diff(arr)[-1] / 2])

    Xp, Yp = np.meshgrid(bin_array(x), bin_array(y))

    return plt.pcolormesh(Xp, Yp, C, *args, **kwargs)


if __name__ == "__main__":
    import utilities as tto
