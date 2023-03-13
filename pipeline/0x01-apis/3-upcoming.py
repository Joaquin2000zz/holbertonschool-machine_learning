#!/usr/bin/env python3
"""
displays the lastest launch information from unofficial SpaceX API
format:
<launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)
"""

import requests


if __name__ == '__main__':
    url_base = 'https://api.spacexdata.com/v4/'
    response = requests.get(url_base + 'launches/latest')
    if response.status_code != 200:
        print('entre al if 0')
        exit()
    r_json = response.json()
    launch_name, date = r_json.get('name'), r_json.get('date_utc')
    if not launch_name or not date:
        print('entre al if 1')
        exit()
    rocket_id, launchpad_id = r_json.get('rocket'), r_json.get('launchpad')
    if not rocket_id or not launchpad_id:
        print('entre al if 2')
        exit()
    rocket = requests.get(url_base + 'rockets/' + rocket_id)
    launchpad = requests.get(url_base + 'launchpads/' + launchpad_id)
    if rocket.status_code != 200 or launchpad.status_code != 200:
        print('entre al if 3')
        exit()
    rocket, launchpad = rocket.json(), launchpad.json()
    if not rocket or not launchpad:
        print('entre al if 4')
        exit()
    rocket_name, launchpad_name = rocket.get('name'), launchpad.get('name')
    launchpad_locality = launchpad.get('locality')
    msg = "{} ({}) {} - {} ({})"
    print(msg.format(launch_name,
                     date, rocket_name,
                     launchpad_name, launchpad_locality))
