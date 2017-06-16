""" Example pipeline for detecting anomalies using an sklearn random forest model
"""
from anomalous.utils import *
from sklearn.ensemble import RandomForestClassifier

df = read_csv(DEFAULT_DB_CSV_PATH)
df = clean_dd_all(df)
X = df.values
anoms = is_anomalous(df)
Y = anoms.values

rf = RandomForestClassifier(max_depth=14, class_weight='balanced', n_estimators=100, n_jobs=-1, warm_start=True)
Y = anoms.values
rf = rf.fit(X, Y)
Y_pred = rf.predict(X)
Y_pred
# array([[ 0.,  0.,  0.,  0.,  0.,  1.],
#        [ 0.,  0.,  0.,  0.,  0.,  1.],
#        [ 0.,  0.,  0.,  0.,  0.,  1.],
#        ..., 
#        [ 0.,  0.,  1.,  1.,  0.,  1.],
#        [ 0.,  0.,  1.,  1.,  0.,  1.],
#        [ 0.,  0.,  1.,  1.,  0.,  1.]])
# correlation for deep and wide random forest
print(pd.np.diag((Y_pred.T.dot(Y) / Y_pred.T.dot(Y_pred) / Y.T.dot(Y)).round(3)))
#  array([ 1.   ,  1.   ,  1.   ,  1.   ,  0.996,  1.   ])
rf.columns = list(df.columns)
rf.feature_names_ = list(df.columns)
pickle.dump(rf, open(DEFAULT_MODEL_PATH, 'wb'))




# these are realistic results with default rf args and using all the 551 features (needle in haystack):
rf = RandomForestClassifier()
rf = rf.fit(X, Y)
rf.predict(X)
# array([[ 0.,  0.,  0.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.,  0.,  0.],
#        ..., 
#        [ 0.,  0.,  1.,  1.,  0.,  1.],
#        [ 0.,  0.,  1.,  1.,  0.,  1.],
#        [ 0.,  0.,  1.,  1.,  0.,  1.]])
((rf.predict(X) - Y) ** 2).sum() ** .5
# 1.7320508075688772
((rf.predict(X)[:,-1] - Y[:,-1]) ** 2).sum() ** .5
# 0.0
((rf.predict(X)[:,0] - Y[:,0]) ** 2).sum() ** .5
# 1.0
((rf.predict(X)[:,1] - Y[:,1]) ** 2).sum() ** .5
# 1.4142135623730951
((rf.predict(X)[:,2] - Y[:,2]) ** 2).sum() ** .5
# 0.0
((rf.predict(X)[:,3] - Y[:,3]) ** 2).sum() ** .5
# 0.0
((rf.predict(X)[:,4] - Y[:,4]) ** 2).sum() ** .5
# 0.0


# these are the perfect results when the only features in X are the features that were thresholded to produce Y:
# array([[ 1.,  0.,  0.,  0.,  0.,  1.],
#        [ 1.,  0.,  0.,  0.,  0.,  1.],
#        [ 1.,  0.,  0.,  0.,  0.,  1.],
#        ..., 
#        [ 0.,  0.,  0.,  1.,  0.,  1.],
#        [ 0.,  0.,  0.,  1.,  0.,  1.],
#        [ 0.,  0.,  0.,  1.,  0.,  1.]])



err = rf.predict(X) - Y
err = pd.DataFrame(rf.predict(X), columns=anoms.columns, index=anoms.index) - pd.DataFrame(Y, columns=anoms.columns, index=anoms.index)
(err[err.columns[-1]] ** 2).mean() ** .5
# 0.0
err.sum()
# proper.redis.requeue.standard.google__anomaly                0.0
# papi.queue.web_insight__anomaly                              0.0
# workers.us.google.status.901 + workers.us.google__anomaly    0.0
# ex_workers.crawler_nodes.bing__anomaly                       0.0
# redis.net.clients__anomaly                                   0.0
# any_anomaly                                                  0.0
# dtype: float64
anoms.sum()
# proper.redis.requeue.standard.google__anomaly                5673
# papi.queue.web_insight__anomaly                              1208
# workers.us.google.status.901 + workers.us.google__anomaly     460
# ex_workers.crawler_nodes.bing__anomaly                       2321
# redis.net.clients__anomaly                                    636
# any_anomaly                                                  8310
# dtype: int64
anoms.sum() / len(anoms)
# proper.redis.requeue.standard.google__anomaly                0.563524
# papi.queue.web_insight__anomaly                              0.119996
# workers.us.google.status.901 + workers.us.google__anomaly    0.045694
# ex_workers.crawler_nodes.bing__anomaly                       0.230555
# redis.net.clients__anomaly                                   0.063177
# any_anomaly                                                  0.825469
