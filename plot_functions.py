#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pygmt
import numpy as np
import matplotlib.pyplot as plt
from obspy import Trace 
from obspy.signal.invsim import cosine_taper


def plot_topography_pygmt(
        title,
        region=[77.93, 78.18, 49.71, 49.86], # defaults to Degelen
        resolution="01s",
        colour=True,
        arrays=True,
        savefig=None
):
    """
    ToDo
    """
    
    format_title = '+t"%s"' %title

    grid_topo = pygmt.datasets.load_earth_relief(resolution=resolution, region=region)
    
    fig = pygmt.Figure()

    if colour:
        pygmt.makecpt(
            cmap='dem4',
            series="0/2000/100",
            continuous=True,
        )
    else:
        pygmt.makecpt(
            cmap='grayC',
            series="0/2000/100",
            continuous=True,
            reverse=True,
        )
    with pygmt.config(FONT_ANNOT_PRIMARY='15p'):
        fig.grdview(
            grid=grid_topo,
            cmap=True,
            region=region,
            projection="M0/0/12c",
            surftype="i",
            shading=True,
            frame=["af", format_title],
            # Sets the height of the three-dimensional relief at 1.5 centimeters
            zsize="1.5c",
        )

    if colour:
        with pygmt.config(FONT_ANNOT_PRIMARY='15p'):
            #fig.colorbar(frame=["a", 'xf500+l"Elevation [m]"'])
            fig.colorbar(frame="xa500f100")

    if arrays:
        vecstyle = "V0.4c+ea"
        fontstyle = "12p,Helvetica-Bold"
        length = 1.5
        # GBA
        fig.plot(
            x=78.04, y=49.733, style=vecstyle, direction=([181], [length]), pen="2p", fill="black"
        )
        fig.text(
            x=78.039, y=49.73, text="GBA", font=fontstyle, justify="ML", offset="0.2c/0c"
        )
        # YKA
        fig.plot(
            x=78.05, y=49.835, style=vecstyle, direction=([6.3], [length]), pen="2p", fill="black"
        )
        fig.text(
            x=78.05, y=49.84, text="YKA", font=fontstyle, justify="ML", offset="0.2c/0c"
        )
        # WRA
        fig.plot(
            x=78.15, y=49.75, style=vecstyle, direction=([128], [length]), pen="2p", fill="black"
        )
        fig.text(
            x=78.15, y=49.753, text="WRA", font=fontstyle, justify="ML", offset="0.2c/0c"
        )

        #EKA
        fig.plot(
            x=77.98, y=49.82, style=vecstyle, direction=([309.5], [length]), pen="2p", fill="black",
        )
        fig.text(
            x=77.975, y=49.825, text="EKA", font=fontstyle, justify="ML", offset="0.2c/0c"
        )
    if savefig is not None:
        fig.savefig(savefig, dpi=150)
    else:
        return fig


def plot_clusters_over_topography_pygmt(
        topography_figure, #pygmt figure object
        dataframe,
        savefig=None
):
    """
    ToDo
    """
    
    dataframe.label = dataframe.label.astype(dtype="category")
    label_annots = list(dataframe.label.cat.categories)

    for i in range(len(label_annots)):
        label_annots[i] = "cluster" + str(label_annots[i])

    pygmt.makecpt(cmap="cat_sns_default.cpt",
                  series=(dataframe.label.cat.codes.min(), dataframe.label.cat.codes.max(), 1),
                  color_model="+c" + ",".join(label_annots),
                  )

    topography_figure.plot(
        x=dataframe.longitude,
        y=dataframe.latitude,
        style="c0.35c",
        pen="1.5p,white",
        fill=dataframe.label.cat.codes.astype(int),  # Points colored by categorical number code
        cmap=True,  # Use colormap created by makecpt
    )

    with pygmt.config(FONT_ANNOT_PRIMARY='15p'):
        topography_figure.colorbar()

    if savefig is not None:
        topography_figure.savefig(savefig, dpi=150)
    else:
        return topography_figure


def plot_cluster_waveforms(array, nclust, tslearn_dataset, labels, timesteps, savefig):
    
    plt.figure()
    
    for yi in range(nclust):

        plt.subplot(nclust, 1, 1 + yi)
        for xx in tslearn_dataset[labels == yi]:
            plt.plot(timesteps, xx.ravel(), "k-", alpha=.2)
        plt.xlim(0, timesteps[-1])
        plt.ylim(-1.1, 1.1)
        plt.title("Cluster %d" % (yi + 1))
        if yi < nclust - 1:
            plt.tick_params(labelbottom=False)
        if yi == 0:
            plt.text(x=0., y=1.3, s=array)

    plt.xlabel("time [s]")
    plt.tight_layout()
    plt.savefig(savefig, dpi=300)


def plot_cluster_centroids(array, model, timesteps, savefig):
    
    plt.figure()

    for i in np.arange(model.cluster_centers_.shape[0]):
        plt.plot(timesteps, model.cluster_centers_[i, :, 0], label="%i" % (i + 1))
        
    plt.xlim(-0.05, timesteps[-1])
    plt.title(array + " kShape centroids")
    plt.legend(title="cluster")
    plt.xlabel("time [s]")
    plt.tight_layout()
    plt.savefig(savefig, dpi=300)


def plot_cluster_spectrograms(array, df, nclust, timesteps, savefig):

    fig, ax = plt.subplots(nclust+1, 1)
    fig.set_size_inches(6, 15)

    nb_events = df.shape[0]
    
    fft_sum = None

    for i in np.arange(nb_events):

        # Read in the trace

        clustID = int(df.iloc[i, df.columns.get_loc("label")])
        trace = Trace(np.array(df.iloc[i, df.columns.get_loc("time_series")]))
        dt = timesteps[1] - timesteps[0]

        taper_percentage = 0.15
        # define taper window
        taper = cosine_taper(trace.stats.npts, taper_percentage)
        # taper the signal
        dat_taper = (trace.data/abs(trace.max())) * taper

        ny = 1 / (2. * dt) # nyquist
        fft = np.fft.rfft(dat_taper, n=trace.stats.npts)  # FFT to frequency domain
        if fft_sum is None:
            fft_sum = np.zeros((nclust + 1, len(fft)))
            fft_sum[clustID, :] = abs(fft)
        else:
            fft_sum[clustID, :] = fft_sum[clustID, :] + abs(fft)

        fft_sum[nclust, :] = fft_sum[nclust, :] + abs(fft)
        f = np.linspace(0, ny, len(fft))  # frequency axis for plotting len(fft)=trace.stats.npts/2+1

        ax[clustID-1].plot(f, (2/trace.stats.npts) * (abs(fft)), "k-", alpha=.2)
        ax[nclust].plot(f, (2/trace.stats.npts) * (abs(fft)), "k-", alpha=.2)

    # ToDo CORRECT HERE, THE SUMS ARE PLOTTED WRONG!
    for j in np.arange(nclust+1):
        ax[j].set_xlim(0, 6)

        if j == nclust:
            ax[j].plot(f, (2 / trace.stats.npts) * (abs(fft_sum[j, :])/nb_events), "r-", linewidth=3)
            ax[j].set_xlabel('Frequency [Hz]')
            title = '%s ALL WAVEFORMS' %(array)#,t_end-t_start)
            ax[j].set_title(title)

        else:
            nb_events_in_cluster = np.count_nonzero(df["label"]==j)
            print(nb_events_in_cluster)
            ax[j].plot(f, (2 / trace.stats.npts) * (abs(fft_sum[j, :])/nb_events_in_cluster), "r-", linewidth=3)
            title = '%s cluster %d' %(array, j+1)
            ax[j].set_title(title)

        ax[j].set_ylabel('Amplitude')
    
    plt.savefig(savefig, dpi=300)
