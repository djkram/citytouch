import rarfile
import json
import os
from django.contrib.gis.geos import Point, GEOSGeometry
import psycopg2
import datetime
import sys
import re
import sys, getopt
import re
from psycopg2.extensions import AsIs
import nltk
import unicodedata

#stopword list
stopwords = set(nltk.corpus.stopwords.words())
with open('stopwords.txt') as f:
    stopwords = stopwords | set(line.strip() for line in f)

# application specific phrases
with open('phrases.txt') as f:
    phrases_pattern = '|'.join((re.escape(phrase.strip()) for phrase in f))

# word mapping
d = dict()
with open('word_map.txt') as f:
    d = dict((line.strip().split() for line in f))

global FILE, DATABASE, TABLE, USER, con
FILE = ''
DATABASE = ''
TABLE = ''
USER = ''

p = "POLYGON((-9.140625 43.45291889355465,-9.84375 42.94033923363183,-8.96484375 40.979898069620155,-10.0634765625 38.65119833229951,-9.2724609375 37.54457732085582,-9.228515625 36.4566360115962,-6.943359374999999 36.66841891894786,-6.328125 35.85343961959182,-4.7900390625 35.96022296929667,-1.58203125 36.63316209558658,0.615234375 38.5825261593533,0.52734375 39.9434364619742,3.3837890625 41.60722821271717,4.04296875 42.553080288955826,-2.28515625 43.929549935614595,-8.6572265625 43.866218006556394,-9.140625 43.45291889355465))"
spain = GEOSGeometry(p)

global r

def sanitize(text):
    text = text.lower()

    #remove non-ASCII chars
    try:
      text = unicode(text, 'ascii', 'ignore')
    except:
      pass
    
    #remove URLs
    text = re.sub(r'http://[^\s]+', ' ', text, flags=re.U)
    #remove hash tags
    text = re.sub(r'#[^\s]+', ' ', text, flags=re.U)
    #remove user mentions
    text = re.sub(r'@[^\s]+', ' ', text, flags=re.U)
    #remove non-word characters
    text = re.sub(r'[^\w\']', ' ', text, flags=re.U)
    #remove RTL characters
    text = re.sub(r'\p{R}', ' ', text, flags=re.U)
    #remove times
    text = re.sub(r'\d{1,2}(?:[.:]\d{2})?\s*(?:[pa]\.?m\.?)?', ' ', text, flags=re.U)
    #remove numbers
    text = re.sub(r'\d+', ' ', text, flags=re.U)
    # remove phrases
    text = re.sub(phrases_pattern, ' ', text, flags=re.U)
    #collapse white space
    text = re.sub(r'\s+', ' ', text, flags=re.U)
    #remove leading, trailing ws
    text = text.strip()

    return [d.get(w,w) for w in text.split(' ') if w not in stopwords and len(w) > 2]

def remove_duplicates(TABLE):
  cur = con.cursor()
  cur.execute("""SET temp_buffers = '500MB';
    CREATE TEMPORARY TABLE t_tmp ON COMMIT DROP AS
    SELECT DISTINCT * FROM %s;
    TRUNCATE %s;
    INSERT INTO %s
    SELECT * FROM t_tmp;""", (AsIs(TABLE), AsIs(TABLE), AsIs(TABLE)))
  con.commit()
  cur.close()

def create_table(TABLE):
  cur = con.cursor()
  cur.execute("SELECT * FROM information_schema.tables WHERE table_name=%s", (TABLE,))
  if not bool(cur.rowcount):
    cur.execute("""CREATE TABLE %s
      (
        id_str text,
        text text,
        created_at numeric,
        geom geometry,
        sanitised text,
        hashtags text,
        mainland boolean,
        user_id text,
        profile_image_url text,
        html text,
        screen_name text
      )
      WITH (
        OIDS=FALSE
      );
      ALTER TABLE %s
        OWNER TO %s;
      
      -- Index: created_at_%s_index
      
      -- DROP INDEX created_at_%s_index;
      
      CREATE INDEX created_at_%s_index
        ON %s
        USING btree
        (created_at );
      
      CREATE INDEX geom_%s_index
        ON %s
        USING gist
        (geom );
  """, (AsIs(TABLE), AsIs(TABLE), AsIs(USER), AsIs(TABLE), AsIs(TABLE), AsIs(TABLE), AsIs(TABLE), AsIs(TABLE), AsIs(TABLE)))
  con.commit()
  cur.close()
  
def processFile(filename):
  cur = con.cursor()
  rf = rarfile.RarFile(filename)
  index = 0
  for arc in rf.infolist():
    if not arc.isdir():
      openf = rf.open(arc.filename)
      for line in openf:
        line = line.strip()
        if line == '':
          continue
        try:
          # Create a dictionary from the line.
          d = json.loads(line)
          
          # We only care about geo-tagged tweets.
          if d['coordinates'] != None and d['coordinates']['type'] == 'Point':
            # Unique id that identifies a tweet.
            id_str = d['id_str']
            
            # Raw and processed versions of the tweet.
            text = d['text']
            hashtags = ','.join(list(set(re.findall(r"#(\w+)", text.lower()))))
            sanitized = ','.join(sanitize(text))
            html = r.sub(r'<a target="_blank" href="\1">\1</a>', text)
            
            # User information
            user_id = d['user']['id']
            profile_image_url = d['user']['profile_image_url']
            screen_name = d['user']['screen_name']
            
            # Create a unix timestamp.
            created_at = reduce(lambda x, y: '%s %s' % (x,  y), filter(lambda x: not x.startswith('+'), d['created_at'].split(' ')))
            created_at = datetime.datetime.strptime(created_at, '%a %b %d %H:%M:%S %Y')
            created_at = int(created_at.strftime('%s')) * 1000
            
            # Create a GEOS Point from the lat, lng.
            lng, lat = d['coordinates']['coordinates']
            p = Point(lng, lat)
            p.set_srid(4326)
            
            # Does this point fall within mainland spain?
            mainland = spain.contains(p)
            
            cur.execute("""INSERT INTO %s(id_str, text, sanitised, hashtags, mainland, created_at, geom, user_id, profile_image_url, html, screen_name) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""", (AsIs(TABLE), id_str, text, sanitized, hashtags, mainland, created_at, p.hex, user_id, profile_image_url, html, screen_name))
            con.commit()
            index += 1
            sys.stdout.write("%i\r" % index)
            sys.stdout.flush()
        except Exception, e:
          print '\tWarning: %s' % str(e)
          pass
    cur.close()
  
if __name__ == "__main__":
  try:
    opts, args = getopt.getopt(sys.argv[1:], "hf:d:t:u:", ["file=", "database=", "table=", "user="])
  except getopt.GetoptError:
    print '%s -f file -d database -t table -u user' % (sys.argv[0])
    sys.exit(2)
  for opt, arg in opts:
    opt = opt.lower()
    if opt in ('-f', '-file'):
       FILE = arg
    elif opt in ('-d', '-database'):
       DATABASE = arg
    elif opt in ('-t', '-table'):
       TABLE = arg
    elif opt in ('-u', '-user'):
       USER = arg
  if DATABASE == '' or TABLE == '' or USER == '':
    print '%s -f file -d database -t table -u user' % (sys.argv[0])
  else:
    con = psycopg2.connect(database=DATABASE, user=USER)
    r = re.compile(r"(http://[^ ]+)")
    print '\nCreating table: %s' % TABLE
    create_table(TABLE)
    print 'Done'
    print 'Processing file(s)'
    if os.path.isdir(FILE):
      for f in os.listdir(FILE):
        if f.endswith('.rar'):
          processFile('%s%s' % (FILE, f))
    else:
      if FILE.endswith('.rar'):
        processFile(FILE)
      else:
        print 'Expected a .rar archive'
    print 'Done'
    print 'Removing duplicates from: %s' % TABLE
    remove_duplicates(TABLE)
    print 'Done'
