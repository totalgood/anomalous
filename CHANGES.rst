=========
Changelog
=========

Version 0.1
===========

- v0.0.1 pip-installable skeleton app
- v0.0.2 function to parse json and extract a time series returning a pd.DataFrame or pd.Series
- v0.0.3 function to parse a directory of json files for a single server and extract all of them to a single pd.DataFrame, dealing with overlapping/conflicting timestamps appropriatelyo
- v0.0.4 function to parse a directory tree of json files and produce one DataFrame for each server (directory)
- v0.0.5 function to tag a time series with anomalous time period
- v0.0.6 tag time series with at least 3 anomalies from the screen shots
- v0.0.7 plots of tagged time series with visual indication of anomalies
 -v0.0.8 confirm plots substantively match the screen shots
- v0.1.0 commandline tool to tag server timeseries with anomalous time periods and trigger processing of a directory tree of json

Version 0.2
===========

- generate binary datetim features (e.g. is_weekday, is holliday, is_monday, is_tuesday, ... is_may, ... is_q1, is_daytime, is_workdaytime)
- generate monitor "rolling window" features: exponential_moving_averages with various alpha/beta weights, std, mean, median, diff with one of the moving averages
- MinMaxNormalize all features 0-1 
- train an isolation forest
- label all the available screenshots of anomalies
- cross-validate anomaly detection against labels
- get feedback on plots of isolation forest anomalies

Version 0.3
===========

- incorporate feedback on isolation forest
- train a ​supervised decision tree
- train a supervised random forest
- cross-validate 
- get feedback from Chase
