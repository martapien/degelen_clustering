#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from obspy import read
from tslearn.utils import to_time_series_dataset
import warnings
import csv
import pandas as pd


def get_sac_filenames(
        array,
        data_path,
        data_filename,
        get_latlon=False
):
    """ ToDo

    """

    # Read in events
    path_to_file = data_path + '/%s/' % array + data_filename

    events_fn = []
    events_ID = []

    if get_latlon:
        events_lat = []
        events_lon = []

    with open(path_to_file, 'rt') as csvfile_read:
        data_reader = csv.reader(csvfile_read, delimiter=',')
        skip = 0
        for row in data_reader:
            if skip == 0:
                skip = 100
                continue

            path = data_path + '/%s/' % array
            fn = path + row[2]
            events_fn.append(fn)
            events_ID.append(row[0])
            if get_latlon:
                if float(row[8]) == 0:
                    events_lat.append(float(row[6])) # GPS lat and lon
                    events_lon.append(float(row[7]))
                else:
                    events_lat.append(float(row[8])) # adit end lat and lon
                    events_lon.append(float(row[9]))

    if get_latlon:
        return events_fn, events_ID, events_lat, events_lon
    else:
        return events_fn, events_ID


def sac2trace(
        sacfile,
        starttime,
        endtime,
        lowpass_filter_parameters=None,
        demean=True,
        normalise=None,
        component="Z"
):

    """ Extrcat a single component from a SAC file and convert it to an obspy.Trace object.

    Parameters:
    sacfile (str): path to a single sac file
    starttime (float): start time in seconds, applied to trim the beginning of the time series
    endtime (float): end time in seconds, applied to trim the end of the trace of the time series
    lowpass_filter_parameters (dict): parameters for an obspy lowpass filter; 
            expects dictionary with keys "fmax", "corners" and "passes"; defaults to None, returning an unfiltered time series
    demean (bool): if Ture, the trace.detrend(type='demean') function is applied to the trace; defaults to True
    normalise (str): 'std' normalises by standard deviation, 'max' normalises by absolute maximu; defaults to None, returning an unnormalised time series
    component (str): "Z", "N", or "E". Defauts to "Z".
    
    Returns:
    obspy.Trace object
    """
    
    st_seis = read(sacfile)
    if component == "Z":
        trace = st_seis[0]
    elif component == "N":
        trace = st_seis[1]
    elif component == "E":
        trace = st_seis[2]
    else:
        raise NotImplementedError("component must be Z N or E")

    # Demean
    if demean:
        trace.detrend(type='demean')

    # Filter 
    if lowpass_filter_parameters is not None:
        trace.filter('lowpass', freq=lowpass_filter_parameters["fmax"], corners=lowpass_filter_parameters["corners"],
                      zerophase=lowpass_filter_parameters["passes"])
    # Normalise
    if normalise is not None:
        if normalise == 'std':
            trace.data = trace.data / trace.std()
        elif normalise == 'max':
            trace.data = trace.data / abs(trace.max())
        else:
            warnings.warn("Normalisation of traces implemented only for standard deviation ('std') and maximum ('max'). \
            Trace was not normalised.")

    # Cut the seismogram for the specified window
    t = trace.stats.starttime
    trace.trim(t + starttime, t + endtime)

    return trace



def sac2tslearn(
        list_of_sacfiles,
        starttime,
        endtime,
        lowpass_filter_parameters=None,
        demean=True,
        normalise=None,
        component="Z",
        output_times=False
):

    """ Extrcat a single component from a series of SAC files and convert them to a tslearn dataset.

    Parameters:
    list_of_sacfiles (list of str): path to a single sac file
    starttime (float): start time in seconds, applied to trim the beginning of the time series
    endtime (float): end time in seconds, applied to trim the end of the trace of the time series
    lowpass_filter_parameters (dict): parameters for an obspy lowpass filter; 
            expects dictionary with keys "fmax", "corners" and "passes"; defaults to None, returning an unfiltered time series
    demean (bool): if Ture, the trace.detrend(type='demean') function is applied to the trace; defaults to True
    normalise (str): 'std' normalises by standard deviation, 'max' normalises by absolute maximu; defaults to None, returning an unnormalised time series
    component (str): "Z", "N", or "E". Defauts to "Z".
    output_times (bool): if Ture, additionally outputs an array with trace times
    
    Returns:
    tslearn dataset, array with trace times (optional)
   """


    nb_events = len(list_of_sacfiles)

    list_of_waveforms = []
    
    for i in np.arange(nb_events):
        # Read in the trace
        sacfile = list_of_sacfiles[i]

        # convert to obspy.trace object
        trace = sac2trace(sacfile, 
                          starttime, 
                          endtime, 
                          lowpass_filter_parameters=lowpass_filter_parameters, 
                          component=component, 
                          demean=demean, 
                          normalise=normalise)
        
        list_of_waveforms.append(trace.data)

    if output_times:
        return to_time_series_dataset(list_of_waveforms), trace.times()
    else:
        return to_time_series_dataset(list_of_waveforms)


def sac2dataframe(
        array_name,
        data_path,
        starttime,
        endtime,
        lowpass_filter_parameters=None,
        demean=True,
        normalise=None,
        component="Z",
        output_times=False
):

    """ Extrcat a single component from a series of SAC files and convert them to a pandas DataFrame.

    Parameters:
    array_name (str): array name, GBA, EKA, YKA or WRA
    data_path (str): path to data
    starttime (float): start time in seconds, applied to trim the beginning of the time series
    endtime (float): end time in seconds, applied to trim the end of the trace of the time series
    lowpass_filter_parameters (dict): parameters for an obspy lowpass filter; 
            expects dictionary with keys "fmax", "corners" and "passes"; defaults to None, returning an unfiltered time series
    demean (bool): if Ture, the trace.detrend(type='demean') function is applied to the trace; defaults to True
    normalise (str): 'std' normalises by standard deviation, 'max' normalises by absolute maximu; defaults to None, returning an unnormalised time series
    component (str): "Z", "N", or "E". Defauts to "Z".
    output_times (bool): if Ture, additionally outputs an array with trace times
    
    Returns:
    pandas DataFrame, array with trace times (optional)
    """
    
    # here we pull the information about the filenames of the events we consider from a CSV file
    # the event ID is the simple numerical ID I have given each event
    list_of_sacfiles, events_ID, events_lat, events_lon = get_sac_filenames(
        array_name,
        data_path,
        "DATA_%s_manual-filter.csv" %array_name, # careful this is hard coded!
        get_latlon=True
    )

    # here match the actual ID that is the date-time-format 
    events_ID_sacfile = []
    for i in np.arange(len(list_of_sacfiles)):
        events_ID_sacfile.append(list_of_sacfiles[i][-23:-12]) # hard coded with respect to existing SAC filenames

    
    nb_events = len(list_of_sacfiles)

    list_of_waveforms = []
    
    for i in np.arange(nb_events):
        # Read in the trace
        sacfile = list_of_sacfiles[i]

        # convert to obspy.trace object
        trace = sac2trace(sacfile, 
                          starttime, 
                          endtime, 
                          lowpass_filter_parameters=lowpass_filter_parameters, 
                          component=component, 
                          demean=demean, 
                          normalise=normalise)
        
        list_of_waveforms.append(trace.data)

    data_array = {"longitude" : events_lon,
                  "latitude" : events_lat,
                  "eventID": events_ID,
                  "date_time": events_ID_sacfile,
                  "time_series": list_of_waveforms
                  }

    data_array["date_time"] = pd.to_datetime(data_array["date_time"])
    #data_array["streamlit_colour"] = '#000000'
    
    df = pd.DataFrame(data=data_array)

    if output_times:
        return df, trace.times()
    else:
        return df

