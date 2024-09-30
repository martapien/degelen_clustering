With this app you can explore the results presented in "Ringing mountain ranges: Teleseismic signature of the interaction of high-frequency wavefields with near-source topography at the Degelen nuclear test site" by Pienkowska et al ([preprint](https://eartharxiv.org/repository/view/7180/)).

Currently the app works only locally. We are working towards deployment on the Streamlit Community Cloud.

To run the app locally, pull this repository and create an environment with `conda` or `mamba` using the provided `environment.yml` file.

```shell
conda env create -f environment.yml
```

Activate the environment and [install Streamlit](https://docs.streamlit.io/get-started/installation/command-line):

```
pip install streamlit
```

Now enter the root of the repo and run:

```
streamlit run degelen_app.py
```

The Degelen clustering app should appear in a new tab in your browser!

In the "Degelen time series" tab you can view the time series and the location of the given event in the mountain range. In the "Degelen clustering" tab you can run the clustering for a selected set of parameters. In the "Results overview" tab you can load images from all of the previous clustering experiments and compare them.

Before you perform the clustering, you can:
- select the array of interest (GBA, YKA, WRA or EKA)
- set the maximum filter frequency for the waveforms (0.5-8 Hz)
- set the length of the time series (2-60 s),
- select the clustering algorithm (kShape or GAK kernel kMeans, see https://tslearn.readthedocs.io/en/latest/gen_modules/tslearn.clustering.html)
- set the number of clusters
- adjust more advanced parameters of the selected clustering algorithm


