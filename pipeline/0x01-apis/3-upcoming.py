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
    r_json = response.json()
    response_sorted = sorted(r_json,
                             key=lambda x: x.get('date_unix'),
                             reverse=True)
    choose = response_sorted[0]

    date = choose.get('date_local')
    launch_name = choose.get('name')

    rocket_id, launchpad_id = choose.get('rocket'), choose.get('launchpad')
    dateu = choose.get('date_unix')

    rocket = requests.get(url_base + 'rockets/' + rocket_id)
    launchpad = requests.get(url_base + 'launchpads/' + launchpad_id)
    rocket, launchpad = rocket.json(), launchpad.json()
    rocket_name, launchpad_name = rocket.get('name'), launchpad.get('name')
    launchpad_locality = launchpad.get('locality')

    msg = "{} ({}) {} - {} ({})"
    print(msg.format(launch_name,
                     date, rocket_name,
                     launchpad_name, launchpad_locality))
