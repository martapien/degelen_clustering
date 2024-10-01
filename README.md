Please note that currently the app works only locally. We are working towards deployment on the Streamlit Community Cloud.

With this app you can explore the results presented in "Ringing mountain ranges: Teleseismic signature of the interaction of high-frequency wavefields with near-source topography at the Degelen nuclear test site" by Pienkowska et al ([preprint](https://eartharxiv.org/repository/view/7180/)). The Degelen time series data come from [AWE Blacknest](https://bdsweb.blacknest.gov.uk/digitised), in particular we cluster some of the [Kazakhstan historic digitised data](https://bdsweb.blacknest.gov.uk/digitised/kazakh.tar.gz)

To run the app locally, pull this repository and create an environment with `mamba` or `conda` using the provided `environment.yml` file. 

(If you need help installing `mamba` or `conda`, see [here](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) or [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)).

```
mamba env create -f environment.yml
```

The default environment name is `degelen_clustering` - you can change this if you like.

Activate the environment and [install Streamlit](https://docs.streamlit.io/get-started/installation/command-line):

```
pip install streamlit
```

Now you need to populate the `data` repository with the Degelen data [that you have to download yourself here](https://bdsweb.blacknest.gov.uk/digitised/kazakh.tar.gz).
Unpack the downoaded `kazakh.tar.gz` and navigate to `kazakh/kazakh/degelen`. The events are sorted by year and date. Now extract the SAC files into the `data` folder as follows:

```commandline
find . -name \*EKA-sum.sac -exec cp {} /path/to/subfolder/data/EKA \;
find . -name \*GBA-sum.sac -exec cp {} /path/to/subfolder/data/GBA \;
find . -name \*YKA-sum.sac -exec cp {} /path/to/subfolder/data/YKA \;
find . -name \*WRA-sum.sac -exec cp {} /path/to/subfolder/data/WRA \;
```

Now enter the root of the repo again and run:

```
streamlit run degelen_app.py
```

The Degelen clustering app should appear in a new tab in your browser!

In the "Degelen time series" tab you can view the time series and the location of the given event in the mountain range. In the "Degelen clustering" tab you can run the clustering for a selected set of parameters. In the "Results overview" tab you can load images from all of the previous clustering experiments and compare them.

Before you perform the clustering, you can:
- select the array of interest (GBA, YKA, WRA or EKA)
- set the maximum filter frequency for the waveforms (0.5-8 Hz)
- set the length of the time series (2-60 s),
- select the clustering algorithm (kShape or GAK kernel kMeans, see [here](https://tslearn.readthedocs.io/en/latest/gen_modules/tslearn.clustering.html)) 
- set the number of clusters
- adjust more advanced parameters of the selected clustering algorithm


