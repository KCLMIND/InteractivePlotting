This is a simple streamlit / python based plotter of data. 

It fits a linear / quadratic / cubic fit to the main data, estimates the residuals, fits a line to that, and from there you get a quick growth chart. 

The options are fairly self explanatory. There are often lots of outliers in imaging data, due to failures in analysis pipelines, atypical anatomy, poor QC or all of these. To address this, I include an option for two-pass fitting - first fit the curve to all data, find subjects who fall outside the normal range (set by you, Z=4 seems fine) and fit again.

The python packages needed are fairly minimal:
streamlit
pandas
numpy
statsmodels
scipy
plotly
os 

When you have installed these, just type in:
streamlit run InteractiveCentilesRobust.py

