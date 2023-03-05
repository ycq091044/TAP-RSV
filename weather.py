# importing library
import requests
from datetime import datetime, timedelta
import time

cur = datetime.now()
cur_date = str(cur)[:10]
while cur_date != "2018-01-01":
    # query and dump
    url = f"https://mesonet.agron.iastate.edu/geojson/cli.py?dl=1&fmt=csv&dt={cur_date}"
    html = requests.get(url).content
    f = open(f"RSV/data/precipitation/{cur_date}.csv", 'wb')
    f.write(html)
    f.close()
    
    # get yesterday
    cur = cur - timedelta(days=1)
    cur_date = str(cur)[:10]
        
    time.sleep(1)
