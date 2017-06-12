from anomalous.utils import *

db = read_csv(DEFAULT_DB_CSV_PATH)
df = db.groupby([db.index]).mean()

query_thresholds = [
    ('*:proper.redis.requeue.standard.google', 2000),
    ('*:((((((((((((((((((((workers.us.google.status.901+workers.us.google.status.902)', 2),
    ('*:ex_workers.crawler_nodes.bing', 10),
    ('*:papi.queue.web_insight', 110000),
]
anoms = pd.DataFrame(dict([(q, df[q] > t) for (q, t) in query_thresholds]))
anoms = anoms.astype(int)
anoms['num_anomolous_metrics'] = anoms.T.sum().T
anoms.describe()
anoms.sum() / len(anoms)
# *:((((((((((((((((((((workers.us.google.status.901+workers.us.google.status.902)    0.118578
# *:ex_workers.crawler_nodes.bing                                                     0.768203
# *:papi.queue.web_insight                                                            0.096887
# *:proper.redis.requeue.standard.google                                              0.146989
# num_anomolous_metrics                                                               1.130657

