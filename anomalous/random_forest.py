""" Example pipeline for detecting anomalies using an sklearn random forest model
"""
from anomalous.utils import *
from sklearn.ensemble import RandomForestClassifier 
rf = RandomForestClassifier()

df = clean_dd_all()
X = df.values
anoms = is_anomalous(df)
Y = anoms.values
rf = rf.fit(X, Y)
rf.predict(X)
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
