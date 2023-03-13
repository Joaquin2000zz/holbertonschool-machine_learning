#!/usr/bin/env python3
"""
function which contains aviableShips fuction
"""
import requests


def availableShips(passengerCount):
    """
    given the @passengerCount, uses Swapi's API to create a method that
    Returns: the @listShips that can hold a given number of passengers
    """
    if not isinstance(passengerCount, int):
        return []
    url = 'https://swapi-api.hbtn.io/api/starships/'
    response = requests.get(url)
    if not isinstance(response, requests.Response):
        return []
    if response.status_code != 200:
        return []

    d = response.json()
    if not d:
        return []

    r = d.get('results')
    f = [None, 'n/a', 'unknown'] + [str(x) for x in range(0, passengerCount)]
    n, p = 'name', 'passengers'
    listShips = [x.get(n) for x in r if x.get(n) and x.get(p) not in f]
    d = response.json()
    next = d.get('next')

    while response and isinstance(listShips, list) and next:
        response = requests.get(next)
        d = response.json()
        if not d:
            return listShips
        r = d.get('results')
        listShips += [x.get(n) for x in r if x.get(n) and x.get(p) not in f]
        next = d.get('next')

    return listShips
