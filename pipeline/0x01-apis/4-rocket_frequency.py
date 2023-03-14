#!/usr/bin/env python3
"""
displays the lastest launch information from unofficial SpaceX API
format:
<launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)
"""
from datetime import datetime
import requests
import re

if __name__ == '__main__':
    url_base = 'https://api.spacexdata.com/v4/'
    response = requests.get(url_base + 'launches/')
    if response.status_code != 200:
        exit()
    r_list = response.json()
    if isinstance(r_list, list):
        choose = []
        times = {}
        for i, d in enumerate(r_list):
            rocket_id = d.get('rocket')
            if not rocket_id:
                exit()
            rocket = requests.get(url_base + 'rockets/' + rocket_id)
            if rocket.status_code != 200:
                exit()
            rocket = rocket.json()
            if not rocket:
                exit()
            rocket_name = rocket.get('name')
            if not times.get(rocket_name):
                times[rocket_name] = 1
            else:
                times[rocket_name] += 1

    def custom(x):
        """
        function to sort first numerically descending
        and then sort alphabetically ascending
        """
        a = re.search('[0-9]', x)
        a = -int(a.group(0)) if a else float('+inf')
        return a, x

    names = sorted(times.keys(), key=custom)

    for name in names:
        print("{}: {}".format(name, times[name]))
