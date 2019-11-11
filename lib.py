import requests

class NBA_stats_getter():
    
    def __init__(self, url):
        self.url = "https://free-nba.p.rapidapi.com/stats"
        headers = {'x-rapidapi-host': "free-nba.p.rapidapi.com",'x-rapidapi-key':"945070f3fbmsh5c886d1719c818dp1891e5jsn35de5214db8b"}
        
    def stat_getter(self, dates):
        stats = requests.request("GET", url, headers = headers, params = dates)
