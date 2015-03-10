import pico
import psycopg2
import numpy as np
import math
import time
import os
import subprocess
import shutil
import operator
import datetime
import svdd
import mmpp
import tfidf
from scipy.special import gammaln
from sklearn.cluster import DBSCAN
import uuid
import os

ROOT = '/app'
DATABASE = os.getenv('BD_NAME_CITYTOUCH', 'citytouch')
HOSTNAME = os.getenv('BD_HOST_CITYTOUCH', 'bdigitaldb.celqzuwfokoe.eu-west-1.rds.amazonaws.com')
USER = os.getenv('BD_USER_CITYTOUCH', 'citytouch')
PASSWORD = os.getenv('BD_PASSWORD_CITYTOUCH', '')

print('BD_PASSWORD_CITYTOUCH:')
print(os.getenv('BD_PASSWORD_CITYTOUCH'))
print(os.environ.get('BD_PASSWORD_CITYTOUCH'))
print(os.environ['PARAM1'])
  
# Session older than 24 hours will be deleted.
SESSION_LIFESPAN = 60 * 60 * 24

con = psycopg2.connect(database=DATABASE, host=HOSTNAME, user=USER, password=PASSWORD) 

R = 6378137.0

def remove_stale_sessions():
  print '------------------Sessions------------------------'
  for session in os.listdir('%s/sessions' % (ROOT)):
    last_modified = os.path.getmtime('%s/sessions/%s' % (ROOT, session))
    now = time.time()
    age = now - last_modified
    
    # If the session is older than 5 minutes delete it.
    if age > SESSION_LIFESPAN:
      shutil.rmtree('%s/sessions/%s' % (ROOT, session))
      print 'Die: %s: %i' % (session, age)
    else:
      print 'Live: %s: %i' % (session, age)
  print '------------------Sessions------------------------'

def get_session_id():
  if not os.path.exists('%s/sessions' % (ROOT)):
    os.makedirs('%s/sessions' % (ROOT))
  remove_stale_sessions()
  return str(uuid.uuid4())

def cluster_DBSCAN(X, eps, min_samples):
  db = DBSCAN(eps, min_samples).fit(np.array(map(lambda d: np.array([85000*d[0], 112000*d[1]]), X)))
  return db.labels_

def great_circle_distance(i, j):
    """ Great circle distance using the Haversine formula """
    lat_i, lon_i = i
    lat_j, lon_j = j
    dlat = math.radians(lat_j-lat_i)
    dlon = math.radians(lon_j-lon_i)
    a = math.sin(dlat/2.0) * math.sin(dlat/2.0) + math.cos(math.radians(lat_i)) \
        * math.cos(math.radians(lat_j)) * math.sin(dlon/2.0) * math.sin(dlon/2.0)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0-a))
    d = R * c
    return d

def getTweetsSVDD(session_id, dataset, min_time, max_time, wkt, polygon_extent):
  start = time.time()
  test_data = []
  for line in open('%s/sessions/%s/tweets_current_hour.txt' % (ROOT, session_id)):
    tw = line.split(',')[1].strip().replace('#','')
    test_data.append([tw])
  end = time.time()
  print 'Load test data: %.2f' % (end-start)
  kernel = tfidf.tfidf_kernel(test_data)
  kernel_function = kernel.kernel_function_tfidf

  kp_dumb = svdd.svdd(kernel_function, 0, [[' ']], 0, 0, 0, 0)
  kp_dumb.load('%s/sessions/%s/svdd' % (ROOT, session_id))
  
  cur = con.cursor()
  query = """SELECT st_y(geom), st_x(geom), created_at, sanitised, hashtags, html, screen_name, profile_image_url, id_str, st_within(geom, st_GeomFromText('%s')) FROM %s WHERE created_at >= %s AND created_at < %s and mainland=true ORDER BY created_at ASC""" % (wkt, dataset, min_time, max_time)
  cur.execute(query)
  
  start = time.time()
  ##   We now apply the trained classifier on another dataset 
  data = []
  predicted = []
  tweets = []
  hashtags = []
  for row in cur.fetchall():
    data.append(list(row))
    line = '%s,%s' % (row[2], ' '.join(row[3].split(',')))
    # add hashtags to processed tweets strings for SVDD
    if not row[4] == '':
      line = '%s %s' % (line, ' '.join(['#%s' % x for x in row[4].split(',')]))
    
    tweet = [line.split(',')[1].replace('#', '')]
    tweets.append(tweet)
    hashtags.append(row[4])
  
  ## batch prediction by SVDD
  predicted = kp_dumb.query(tweets).flatten()
  
  #pmins = np.min(predicted, axis=0)
  #pmaxs = np.max(predicted, axis=0)
  #rng = pmaxs - pmins
  #predicted = np.max(polygon_extent) - (((np.max(polygon_extent) - np.min(polygon_extent)) * (pmaxs - predicted)) / rng)
  
  
  end = time.time()
  print 'Total query time: %.2f' % (end-start)
  
  # remove hashtags from processed tweets strings for cleaner tag clouds  
  for i, htags in enumerate(hashtags):
    for h in htags.split(','):
      tweets[i][0] = tweets[i][0].replace(h, '')
  words = {}
  for index, tweet in enumerate(tweets):
    for word in tweet[0].split(' '):
       if not word in words:
         words[word] = 0
       words[word] += predicted[index]
  
  word_cloud = [{'text': x[0], 'size': x[1]} for x in sorted(words.iteritems(), key=operator.itemgetter(1), reverse=True)[:50] if x[0] != '' and x[0] != 'barcelona']
  # rescale related to predefined range
  wmax = word_cloud[0]['size']
  wmin = word_cloud[-1]['size']
  rng = wmax - wmin 
  for t in word_cloud:
    t['size'] = np.max(polygon_extent) - (((np.max(polygon_extent) - np.min(polygon_extent)) * (wmax - t['size'])) / rng)
  
  hashs = {}
  for index, hashtag in enumerate(hashtags):
    for h in hashtag.split(','):
       if not h in hashs:
         hashs[h] = 0
       hashs[h] += predicted[index]
  hash_cloud = [{'text': x[0], 'size': x[1]} for x in sorted(hashs.iteritems(), key=operator.itemgetter(1), reverse=True)[:50] if x[0] != '' and x[1] > 0]  
  # rescale related to predefined range
  hmax = hash_cloud[0]['size']
  hmin = hash_cloud[-1]['size']
  rng = hmax - hmin
  for t in hash_cloud:
    t['size'] = np.max(polygon_extent) - (((np.max(polygon_extent) - np.min(polygon_extent)) * (hmax - t['size'])) / rng)
  
  return {'data': data, 'predicted': predicted, 'word_cloud': word_cloud, 'hash_cloud': hash_cloud}

def getTweetsFor(dataset, min_time, max_time, algorithm, eps, min_points):
  start = time.time()
  cur = con.cursor()
  query = """SELECT st_y(geom), st_x(geom), created_at, html, sanitised, hashtags, screen_name, profile_image_url, id_str FROM %s WHERE created_at >= %s AND created_at < %s and mainland=true ORDER BY created_at ASC""" % (dataset, min_time, max_time)
  cur.execute(query)
  data = []
  X = []
  for row in cur.fetchall():
    X.append(np.array([row[0], row[1]]))
    data.append(list(row))
  end = time.time()
  print 'QUERY: %.2f seconds' % (end-start)
  
  if algorithm == 'DBSCAN-SciPy':
    start = time.time()
    labels = cluster_DBSCAN(X, float(eps), int(min_points))
    end = time.time()
  
  # Add support for different algorithms here.
  else:
    pass
  
  print '%s(eps=%s, min_points=%s): %.2f seconds' % (algorithm, eps, min_points, end-start)
  
  start = time.time()
  tweet_clouds = {}
  hash_clouds = {}
  for i, label in enumerate(labels):
    data[i].append(label)
    if not label in tweet_clouds: tweet_clouds[label] = {}
    if not label in hash_clouds: hash_clouds[label] = {}
    for word in data[i][4].strip().split(','):
      if not word in tweet_clouds[label]: tweet_clouds[label][word] = 0
      tweet_clouds[label][word] += 1
    for hash in data[i][5].strip().split(','):
      if not hash in hash_clouds[label]: hash_clouds[label][hash] = 0
      hash_clouds[label][hash] += 1
  tweetClouds = {}
  hashClouds = {}
  for label in tweet_clouds:
    tweetCloud = []
    hashCloud = []
    for word in tweet_clouds[label]:
      if word != "":
        if word != "barcelona":
          tweetCloud.append({'text': word, 'size': tweet_clouds[label][word]})
    tweetClouds[int(label)] = sorted(tweetCloud, key=lambda x: x['size'], reverse=True)[:100]
    for hash in hash_clouds[label]:
      if hash != "":
        hashCloud.append({'text': hash, 'size': hash_clouds[label][hash]})
    hashClouds[int(label)] = sorted(hashCloud, key=lambda x: x['size'], reverse=True)[:100]
  end = time.time()
  print '(TWEET & HASH) CLOUDS: %.2f seconds' % (end-start)
  
  # Remove duplicate points, won't be seen if draw ontop of each other.
  if False:
    start = time.time()
    duplicates = 0
    s = set([])
    for i in reversed(range(len(data))):
      d = data[i]
      tup = (d[0], d[1])
      if not tup in s:
        s.add(tup)
      else:
        duplicates += 1
        del(data[i])
    end = time.time()
    print '%i = %f.2%%: %.2f seconds' % (duplicates, (duplicates / float(len(s))) * 100.0, (end-start))
  return {'tweetClouds': tweetClouds, 'hashClouds': hashClouds, 'data': data}

def get_tweets_for_polygon(session_id, dataset, wkt, g_time):
  cur = con.cursor()
  
  data = cur.execute("""SELECT min(created_at) / 1000, max(created_at) / 1000 FROM %s""" % (dataset, ))
  data = cur.fetchone()
  # Remove the minutes and seconds from first.
  first = datetime.datetime.fromtimestamp(int(data[0]))
  first -= datetime.timedelta(minutes=first.minute)
  first -= datetime.timedelta(seconds=first.second)
  
  # Remove the minutes and seconds from last, then add one hour.
  last = datetime.datetime.fromtimestamp(int(data[-1]))
  last -= datetime.timedelta(minutes=last.minute)
  last -= datetime.timedelta(seconds=last.second)
  last += datetime.timedelta(hours=1)
  
  data = cur.execute("""SELECT created_at / 1000, sanitised FROM %s WHERE st_within(geom, st_GeomFromText('%s')) ORDER BY created_at ASC""" % (dataset, wkt))
  data = cur.fetchall()
  
  if len(data) < 10:
    return [[], [], []]
  
  # Build up buckets by hour.
  hours = {}
  documents = {}
  time = first
  while time <= last:
    hours[time] = 0
    documents[time] = ''
    time += datetime.timedelta(hours=1)

  for d in data:
    time = datetime.datetime.fromtimestamp(d[0])
    time -= datetime.timedelta(minutes=time.minute)
    time -= datetime.timedelta(seconds=time.second)
    if time >= first and time <= last:
      hours[time] += 1
      documents[time] += d[1] + ","
  
  # Tweets within the selected polygon, for the proceeding hour, starting at the selected moment in time.
  tweets_polygon_current_hour = []
  d = cur.execute("""SELECT created_at, sanitised, hashtags FROM %s WHERE created_at >= %s AND created_at < %s AND mainland = true AND st_within(geom, st_GeomFromText('%s')) ORDER BY created_at ASC""" % (dataset, g_time, g_time + (1000 * 60 * 60), wkt))
  d = cur.fetchall()
  for row in d:
    line = '%s,%s' % (row[0], ' '.join(row[1].split(',')))
    if not row[2] == '':
      line = '%s %s' % (line, ' '.join(['#%s' % x for x in row[2].split(',')]))
    tweets_polygon_current_hour.append([line.split(',')[1].replace('#', '')])
  
  # Tweets for all of a Spain, for the proceeding hour, starting at the selected moment in time.
  tweets_current_hour = []
  hashtags = []
  
  if not os.path.exists('%s/sessions/%s/' % (ROOT, session_id)):
    os.makedirs('%s/sessions/%s/' % (ROOT, session_id))
  
  with open('%s/sessions/%s/tweets_current_hour.txt' % (ROOT, session_id), 'w') as f_out:
    d = cur.execute("""SELECT created_at, sanitised, hashtags FROM %s WHERE created_at >= %s AND created_at < %s AND mainland = true ORDER BY created_at ASC""" % (dataset, g_time, g_time + (1000 * 60 * 60)))
    d = cur.fetchall()
    for row in d:
      line = '%s,%s' % (row[0], ' '.join(row[1].split(',')))
      if not row[2] == '':
        line = '%s %s' % (line, ' '.join(['#%s' % x for x in row[2].split(',')]))
      f_out.write('%s\n' % line)
      tweets_current_hour.append([line.split(',')[1].replace('#', '')])
      hashtags.append(row[2])
  
  data = [[int(x.strftime('%s')) * 1000, hours[x]] for x in sorted(hours.keys())]
  values = []
  for d in data:
    values.append(float(d[1]))
  values = np.array(values)
  
  Nh = 24 # number of time intervals in a day
  offset = (4 - first.hour) % Nh  # skip the beginning until 4am
  day_shift = (last - (first + datetime.timedelta(hours=offset))).days % 7 # Shift so that hours are taken from the start instead of the end.
  offset += (Nh * day_shift)
  Nw = ((last - (first + datetime.timedelta(hours=offset))).days / 7) # number of weeks

  values = values[offset:offset+7*Nw*Nh]
  
  valuessc = 100.0*values/np.max(values)
  
  N = valuessc.reshape(7*Nw, Nh)
  event_times = np.zeros(N.shape);


  ## MMPP
  ##
  
  mp = mmpp.mmpp(event_length = 0.5)
  routine, events = mp.train(N, [20,5])
  routine = np.max(values)*routine/100.0
  
  
  ## SVDD
  ##
  # SVDD learning parameters
  maxsize = 150       # dictionary size in samples
  ald_thresh = 0.01   # linear dependence threshold, 0 means every sample is novel
  adaptive = True      # data-adaptive dictionary clean up when maxsize reached
  forget_rate = 0.0     # force to eliminate the oldest entries from time to time 

  num_samples = len(tweets_polygon_current_hour)  # length of the stream
  
  kernel = tfidf.tfidf_kernel(tweets_polygon_current_hour)
  
  targets = np.ones( (num_samples, 1) )
  ##    Initialize the SVDD algorithm
  kp = svdd.svdd(kernel.kernel_function_tfidf, ald_thresh, [tweets_polygon_current_hour[0]], targets[0], maxsize, adaptive, forget_rate)
  
  # train on a set of tweets
  for i in range(1,num_samples): 
    kp.update([tweets_polygon_current_hour[i]], targets[i])

  ##   We now apply the trained classifier on another dataset 
  predicted = [] 
  predicted = kp.query(tweets_current_hour)
  
  ind = np.argsort(predicted.flatten())
  kp.save('%s/sessions/%s/svdd' % (ROOT, session_id))
  
  timeline2 = []
  timeline3 = []
  max_value = np.max(values)
    
  for i, epoch in enumerate(sorted(hours.keys())[offset:offset+7*Nw*Nh]):
    e = int(epoch.strftime('%s')) * 1000
    timeline2.append([e, routine[i]])
    timeline3.append([e, max_value * events[i]])
  
  words = {}
  for index, tweet in enumerate(tweets_current_hour):
    for word in tweet[0].split(' '):
       if not word in words:
         words[word] = 0
       words[word] += predicted[index][0]
  word_cloud = [{'text': x[0], 'size': x[1]} for x in sorted(words.iteritems(), key=operator.itemgetter(1), reverse=True)[:50] if x[0] != 'barcelona']
  
  hashs = {}
  for index, hashtag in enumerate(hashtags):
    for h in hashtag.split(','):
       if not h in hashs:
         hashs[h] = 0
       hashs[h] += predicted[index][0]
  hash_cloud = [{'text': x[0], 'size': x[1]} for x in sorted(hashs.iteritems(), key=operator.itemgetter(1), reverse=True)[:50] if x[0] != '' and x[1] > 0]
  
  return {'min_time': data[0][0], 'max_time': data[-1][0], 'max_value': max_value, 'raw': data, 'routine': timeline2, 'events': timeline3, 'predicted': predicted, 'word_cloud': word_cloud, 'hash_cloud': hash_cloud}
