#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, KShape
from read_sac import get_sac_filenames, sac2tslearn
import traceback
import pandas as pd
import numpy as np
import streamlit as st


def clustering_model(clustering_parameters):
    """
    Defines one of the tslearn models given the provided clustering parameters: 
    the KernelKMeans with Global Alignment Kernel (gak_km) or the
    kShape algorithm based on cross-correlations, see tselarn documentaition, for example
    https://tslearn.readthedocs.io/en/stable/user_guide/clustering.html#k-means-and-dynamic-time-warping

    Parameters:
    clustering_parameters (dict): dictionary with parameters for defining a tslearn model class
    
    Returns:
    tselarn model class
    
    """
    
    if clustering_parameters["clustering"] == "gak_km":

        # if sigma is undefined, set sigma to 1 as the default
        if clustering_parameters["gak_km"]["kernel_params"]["sigma"]:
            kernel_params = clustering_parameters["gak_km"]["kernel_params"]
        
        else:
            kernel_params = {"sigma": 1}
                
        model = KernelKMeans(n_clusters=clustering_parameters["gak_km"]["n_clusters"],
                             max_iter=clustering_parameters["gak_km"]["max_iter"],
                             tol=float(clustering_parameters["gak_km"]["tol"]),
                             kernel="gak", # (default)
                             kernel_params=kernel_params,
                             n_jobs=clustering_parameters["gak_km"]["n_jobs"],
                             n_init=clustering_parameters["gak_km"]["n_init"])
        
    elif clustering_parameters["clustering"] == "ks":
        model = KShape(n_clusters=clustering_parameters["ks"]["n_clusters"],
                       max_iter=clustering_parameters["ks"]["max_iter"],
                       tol=float(clustering_parameters["ks"]["tol"]),
                       n_init=clustering_parameters["ks"]["n_init"])
        
    else:
        raise NotImplementedError("The implemented algorithms inclue GAK lernel k-means (gak_km) and kShape (ks).")
                
    return model

@st.cache_data
def perform_clustering(
    array_name,
    tslearn_dataset,
    clustering_parameters
):
    """
    # create subfolders
    #if not os.path.exists(outputs_path + "/%s" % array_name):
    #    os.makedirs(outputs_path + "/%s" % array_name)

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

    # here we take data from SAC files, filter, cut, normalise, demean
    # and output a tslearn dataset object
    tslearn_dataset, timesteps = sac2tslearn(
        list_of_sacfiles,
        starttime,
        endtime,
        lowpass_filter_parameters=lowpass_filter_parameters,
        normalise=normalise,
        output_times=True
        )

    """
    model = clustering_model(clustering_parameters)

    # here we do the clustering; sometimes it fails for one of the trials (the nb of
    # trials is the n_init), hence the try-except; if that happens just re-try
    while True:
        try:
            labels = model.fit_predict(tslearn_dataset)
        except:
            print("Error in model.fit_predict() for %s.  Trying again." %array_name)
            traceback.print_exc()
        else: 
            print("Successful model.fit_predict() for %s." %array_name)
            break

    # here we re-label the clusters, as the labels go from 0 and we want them from 1
    # and then create a pandas dataframe with a summary of the clusters
    text_labels = []
    for i in np.arange(len(labels)):
          text_labels.append(str(labels[i]+1))
    """
    data_array = {"longitude" : events_lon,
                  "latitude" : events_lat,
                  "label": text_labels,
                  "eventID": events_ID,
                  "originalID": events_ID_sacfile,
                  "time_series": [*tslearn_dataset[:,:,0]]
                  }
    df = pd.DataFrame(data=data_array)
    """
    # save the actual model object, so that we can read it later
    # if clustering_parameters["clustering"] == "km":
    #     model.to_hdf5(tslearn_path + "/%s/km_model.hdf5" % array_name)
    # if clustering_parameters["clustering"] == "gak_km":
    #     model.to_hdf5(tslearn_path + "/%s/gak_km_model.hdf5" % array_name)
    # if clustering_parameters["clustering"] == "ks":
    #     model.to_hdf5(tslearn_path + "/%s/ks_model.hdf5" % array_name)

    # # save the clusters summary (with IDs of events and their lat-lon info) into CSV
    # df.to_csv(tslearn_path + "/%s/data.csv" % array_name)
    
    return text_labels, model
    
    

def compute_sigma_GAK(model, clustering_parameters):
    """
    ToDo look at this
    Not sure if this is useful at all, not used in the current test
    """
    
    # here if the sigma: "" for gak_km, we compute it given the n_samples and random_state
    # specified in the parameters file using the sigma_gak method from tselarn
    # it prints the sigma for each array to file
    estimate_sigma = sigma_gak(
        dataset=tslearn_dataset,
        n_samples=clustering_parameters["gak_km"]["estimate_sigma_n_samples"],
        random_state=clustering_parameters["gak_km"]["estimate_sigma_random_state"]
    )
    # print("sigma for %s: "%array, estimate_sigma)
    kernel_params = {"sigma": estimate_sigma}
    model.set_params(kernel_params=kernel_params)

    return model
