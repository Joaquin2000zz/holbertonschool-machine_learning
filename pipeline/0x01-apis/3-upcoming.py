#!/usr/bin/env python3
"""
displays the lastest launch information from unofficial SpaceX API
format:
<launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)
"""
import requests


if __name__ == '__main__':
    url_base = 'https://api.spacexdata.com/v4/'
    response = requests.get(url_base + 'launches/upcoming')
    if response.status_code != 200:
        exit()
    r_json = response.json()
    response_sorted = sorted(r_json,
                             key=lambda x: x.get('date_unix'),
                             reverse=True)
    choose = response_sorted[0]

    date = choose.get('date_local')
    launch_name = choose.get('name')
    if not launch_name or not date:
        exit()

    rocket_id, launchpad_id = choose.get('rocket'), choose.get('launchpad')
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
