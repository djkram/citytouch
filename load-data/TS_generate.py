import json
import datetime
import psycopg2
import sys, getopt

global DATABASE, TABLE, USER, con
DATABASE = ''
TABLE = ''
USER = ''

if __name__ == "__main__":
  try:
    opts, args = getopt.getopt(sys.argv[1:], "hd:t:u:", ["database=", "table=", "user="])
  except getopt.GetoptError:
    print '%s -d database -t table -u user' % (sys.argv[0])
    sys.exit(2)
  for opt, arg in opts:
    opt = opt.lower()
    if opt in ('-d', '-database'):
       DATABASE = arg
    elif opt in ('-t', '-table'):
       TABLE = arg
    elif opt in ('-u', '-user'):
       USER = arg
  if DATABASE == '' or TABLE == '' or USER == '':
    print '%s -d database -t table -u user' % (sys.argv[0])
  else:
    con = psycopg2.connect(database=DATABASE, user=USER) 
    cur = con.cursor()
    cur.execute('select created_at / 1000 from %s order by created_at' % (TABLE,))
    data = cur.fetchall()

    # Remove the minutes and seconds from first.
    first = datetime.datetime.fromtimestamp(int(data[0][0]))
    first -= datetime.timedelta(minutes=first.minute)
    first -= datetime.timedelta(seconds=first.second)

    # Remove the minutes and seconds from last, then add one hour.
    last = datetime.datetime.fromtimestamp(int(data[-1][0]))
    last -= datetime.timedelta(minutes=last.minute)
    last -= datetime.timedelta(seconds=last.second)
    last += datetime.timedelta(hours=1)

    # Build up buckets by hour.
    hours = {}
    time = first
    while time <= last:
      hours[time] = 0
      time += datetime.timedelta(hours=1)

    for d in data:
      t = datetime.datetime.fromtimestamp(d[0])
      t -= datetime.timedelta(minutes=t.minute)
      t -= datetime.timedelta(seconds=t.second)
      hours[t] += 1

    timeseries = [[int(x.strftime('%s'))*1000, hours[x]] for x in sorted(hours.keys())]
    json.dump({'timeseries': timeseries, 'min_time': timeseries[0][0], 'max_time': timeseries[-1][0]}, open('%s.json' % (TABLE,), 'w'))
