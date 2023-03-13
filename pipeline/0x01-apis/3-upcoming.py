#!/usr/bin/env python3
"""
displays the lastest launch information from unofficial SpaceX API
format:
<launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)
"""
from datetime import datetime
import requests


if __name__ == '__main__':
    url_base = 'https://api.spacexdata.com/v4/'
    response = requests.get(url_base + 'launches')
    if response.status_code != 200:
        exit()
    r_json = response.json()
    if isinstance(r_json, list):
        choose = []
        indexes = {}
        for i, d in enumerate(r_json):
            date = d.get('date_unix')
            date = datetime.utcfromtimestamp(date).strftime(
                '%Y-%m-%dT%H:%M:%S.%f%z'
                )
            indexes[date] = i
            choose.append(date)
            choose.sort()
        date = choose[::-1][0]
        r_json = r_json[indexes.get(date)]

    launch_name, date = r_json.get('name'), r_json.get('date_unix')
    if not launch_name or not date:
        exit()
    date = datetime.utcfromtimestamp(date).strftime('%Y-%m-%dT%H:%M:%S.%f%z')
    rocket_id, launchpad_id = r_json.get('rocket'), r_json.get('launchpad')
    if not rocket_id or not launchpad_id:
        exit()
    rocket = requests.get(url_base + 'rockets/' + rocket_id)
    launchpad = requests.get(url_base + 'launchpads/' + launchpad_id)
    if rocket.status_code != 200 or launchpad.status_code != 200:
        exit()
    rocket, launchpad = rocket.json(), launchpad.json()
    if not rocket or not launchpad:
        exit()
    rocket_name, launchpad_name = rocket.get('name'), launchpad.get('name')
    launchpad_locality = launchpad.get('locality')
    msg = "{} ({}) {} - {} ({})"
    print(msg.format(launch_name,
                     date, rocket_name,
                     launchpad_name, launchpad_locality))
