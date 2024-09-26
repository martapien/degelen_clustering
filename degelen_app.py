import streamlit as st
import numpy as np
import pandas as pd
import yaml
from read_sac import sac2dataframe
from clustering import perform_clustering
from plot_functions import plot_topography_pygmt, plot_clusters_over_topography_pygmt, plot_cluster_waveforms, plot_cluster_centroids, plot_cluster_spectrograms
from tslearn.utils import to_time_series_dataset
import matplotlib.pyplot as plt
import glob


st.title("Degelen: waveforms clustering")

# parameters: hard-coded
starttime = 19.25
data_path = "./data/"
outputs_path = "./outputs/"

st.sidebar.write("**SETTINGS: pre-processing of time series:**")

# parameters: selected in the app
select_array = st.sidebar.selectbox(
    'Select the array of interest:',
     ["GBA", "EKA", "YKA", "WRA"])

select_fmax = st.sidebar.slider(label='set maximum filter frequency [Hz]', min_value=0.5, max_value=8., value=2., step=0.5)  # min: 0.5Hz, max: 8Hz default: 2Hz
select_window_length = st.sidebar.slider(label='set length of time series [s]', min_value=2, max_value=60, value=20, step=1)  
select_normalise = st.sidebar.selectbox(
    'Select how to normalise time series:',
     ["max", "std", None])


endtime = select_window_length + starttime

# here some are selected in the app, some are hard-coded
filter_parameters = {
    "fmax" : select_fmax,
    "corners": 4,
    "passes": True,
}

# here some are selected in the app, some are hard-coded
clustering_parameters_file = "clustering_parameters.yaml"
with open(clustering_parameters_file) as f:
    clustering_parameters = yaml.full_load(f)

st.sidebar.write("**SETTINGS: clustering algorithms:**")

select_algorithm = st.sidebar.selectbox(
    'Select clustering algorithm:',
     ["kShape", "GAK_kernel_kMeans"])
select_nclust = st.sidebar.slider(label='set the number of clusters', min_value=2, max_value=6, value=3, step=1)  

select_clust_params = st.sidebar.toggle("Would you like to modify more advanced parameters of the clustering algorithm?")

    
if select_algorithm == "kShape":
    clustering_parameters["clustering"] = 'ks'
    clustering_parameters['ks']['n_clusters'] = select_nclust
    if select_clust_params:
        st.sidebar.write("For details on the parameters, see [tslearn documentation](https://tslearn.readthedocs.io/en/latest/gen_modules/clustering/tslearn.clustering.KShape.html).")
        clustering_parameters['ks']['max_iter'] = st.sidebar.number_input(label='max_iter', value=50)  
        clustering_parameters['ks']['tol'] = st.sidebar.number_input(label='tol', value=1e-7, format="%f")   
        clustering_parameters['ks']['n_init'] = st.sidebar.number_input(label='n_init', value=50)

else:
    clustering_parameters["clustering"] = 'gak_km'
    clustering_parameters['gak_km']['n_clusters'] = select_nclust
    if select_clust_params:
        st.sidebar.write("For details on the parameters, see [tslearn documentation](https://tslearn.readthedocs.io/en/latest/gen_modules/clustering/tslearn.clustering.KernelKMeans.html).")
        clustering_parameters['gak_km']['max_iter'] = st.sidebar.number_input(label='max_iter', value=50)  
        clustering_parameters['gak_km']['tol'] = st.sidebar.number_input(label='tol', value=1e-10, format="%f")  
        clustering_parameters['gak_km']['n_init'] = st.sidebar.number_input(label='n_init', value=30)
        clustering_parameters['gak_km']['n_jobs'] = st.sidebar.number_input(label='n_jobs (careful, see how many processors you have!)', value=4, min_value=1, max_value=8)
        
# column configurations to display the data
column_configuration = {
    "longitude": st.column_config.NumberColumn(
        "longitude", help="The longitude coordinates of the explosion.", 
    ),
    "latitude": st.column_config.NumberColumn(
        "latitude", help="The latitude coordinates of the explosion.", 
    ),
    "eventID": st.column_config.TextColumn(
        "event ID",
        help="Event ID of the explosion as in Pienkowska et al. (2024).",
    ),
    "date_time": st.column_config.DatetimeColumn(
        "date and time",
        help="The date and time of the explosion.",
    ),
    "time_series": st.column_config.LineChartColumn(
        "time series",
        help="The time series of the explosion (normalised if normalisaion selected).",
        width="large",
    ),
}

# column configuration to display the data
column_configuration2 = {
    "label": st.column_config.NumberColumn(
        "cluster", help="The label of the cluster associated with the explosion.", 
    ),
    "longitude": st.column_config.NumberColumn(
        "longitude", help="The longitude coordinates of the explosion.", 
    ),
    "latitude": st.column_config.NumberColumn(
        "latitude", help="The latitude coordinates of the explosion.", 
    ),
    "eventID": st.column_config.TextColumn(
        "event ID",
        help="Event ID of the explosion as in Pienkowska et al. (2024).",
    ),
    "date_time": st.column_config.DatetimeColumn(
        "date and time",
        help="The date and time of the explosion.",
    ),
    "time_series": st.column_config.LineChartColumn(
        "time series",
        help="The time series of the explosion (normalised if normalisaion selected).",
        width="large",
    ),
}

df_GBA, times_GBA = sac2dataframe(
        "GBA",
        data_path,
        starttime,
        endtime,
        lowpass_filter_parameters=filter_parameters,
        demean=True,
        normalise=select_normalise,
        output_times=True
)

df_EKA = sac2dataframe(
        "EKA",
        data_path,
        starttime,
        endtime,
        lowpass_filter_parameters=filter_parameters,
        demean=True,
        normalise=select_normalise
)

df_YKA = sac2dataframe(
        "YKA",
        data_path,
        starttime,
        endtime,
        lowpass_filter_parameters=filter_parameters,
        demean=True,
        normalise=select_normalise
)

df_WRA = sac2dataframe(
        "WRA",
        data_path,
        starttime,
        endtime,
        lowpass_filter_parameters=filter_parameters,
        demean=True,
        normalise=select_normalise
)

if select_array == "GBA":
    df = df_GBA
elif select_array == "EKA":
    df = df_EKA    
elif select_array == "YKA":
    df = df_YKA
else:
    df = df_WRA    

data, clustering, overview = st.tabs(["Degelen time series", "Degelen clustering", "Results overview"])

with data: # Add data tab #############################################

    st.header("All Degelen events at %s." %select_array)
    
    st.map(df)
    
    event = st.dataframe(
        df,
        column_config=column_configuration,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
    )
    highlight_events = event.selection.rows    
    
    if highlight_events:
    
        st.header("Selected Degelen events %s." %select_array)
        st.map(df.iloc[highlight_events])
    
        fig, ax = plt.subplots()
        for event in highlight_events:
            label = "event " + str(df.iloc[event, df.columns.get_loc("eventID")]) 
            ax.plot(times_GBA, np.array(df.iloc[event, df.columns.get_loc("time_series")]), label=label)
            ax.set_title("Selected %s waveforms."%select_array)
            plt.legend() 
    
        st.pyplot(fig)
        
with clustering: # Add clustering tab #############################################
    st.write("**Selected algorithm:**", select_algorithm)
    st.write("**Selected array:**", select_array)

    if select_algorithm == "kShape":
        st.write("**Parameters:**", clustering_parameters['ks'])
    else:
        st.write("**Parameters:**", clustering_parameters['gak_km'])

    if st.button('Run clustering'):
        labels, model = perform_clustering(select_array, to_time_series_dataset(df["time_series"]), clustering_parameters)
        df["label"] = labels

        st.dataframe(
            df,
            column_config=column_configuration2,
            use_container_width=True,
            hide_index=True,
    
        )
        
        savefig_path_template = outputs_path + select_array + "_" + select_algorithm + "_fmax{}_len{}_nclust{}_maxiter{}_tol{}_ninit{}".format(
            select_fmax,
            select_window_length,
            select_nclust,
            clustering_parameters['ks']['max_iter'],
            clustering_parameters['ks']['tol'],
            clustering_parameters['ks']['n_init'])

        savefig_path_map = savefig_path_template + "_map.png"
        st.write("Saving ma with clusters to ", savefig_path_map)
        # plot the spatial distribution of the clusters 
        topo_fig = plot_topography_pygmt(select_array, colour=False)
        plot_clusters_over_topography_pygmt(
                topo_fig,  # pygmt figure object
                df,
                savefig=savefig_path_map
        )
    
        st.image(savefig_path_map)

        savefig_path_waveforms = savefig_path_template + "_waveforms.png"
        st.write("Saving cluster waveforms to ", savefig_path_waveforms)
        figure = plot_cluster_waveforms(select_array, select_nclust, to_time_series_dataset(df["time_series"]), model.labels_, times_GBA, savefig_path_waveforms)
        st.image(savefig_path_waveforms)

        if select_algorithm == "kShape":
            savefig_path_centroids = savefig_path_template + "_centroids.png"
            st.write("Saving cluster centroids to ", savefig_path_centroids)
            figure = plot_cluster_centroids(select_array, model, times_GBA, savefig_path_centroids)
            st.image(savefig_path_centroids)

        savefig_path_spectrograms = savefig_path_template + "_spectrograms.png"
        st.write("Saving cluster spectrorgams to ", savefig_path_centroids)
        figure = plot_cluster_spectrograms(select_array, df, select_nclust, times_GBA, savefig_path_spectrograms)
        st.image(savefig_path_spectrograms)
        
with overview: # Add overview tab #############################################
    list_of_files = glob.glob(outputs_path + "*")
    select_display = st.multiselect("Select images to view.", list_of_files) 
    cols = st.columns(2)
    for i in np.arange(len(select_display)):
        if i % 2:
            cols[1].image(select_display[i], caption=select_display[i][10:-4])
        else:
            cols[0].image(select_display[i], caption=select_display[i][10:-4])