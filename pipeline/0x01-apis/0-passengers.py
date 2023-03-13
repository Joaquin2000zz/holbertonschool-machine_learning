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
        return None
    url = 'https://swapi-api.hbtn.io/api/starships/?passengers='
    url += str(passengerCount)
    response = requests.get(url)
    if not isinstance(response, requests.Response):
        raise Exception

    D = response.json()
    r = D.get('results')
    f = [None, 'n/a', 'unknown'] + [str(x) for x in range(0, passengerCount)]
    n, p = 'name', 'passengers'
    listShips = [x.get(n) for x in r if x.get(n) and x.get(p) not in f]
    D = response.json()
    next = D.get('next')

    while response and isinstance(listShips, list):
        if not next:
            return listShips
        response = requests.get(next)
        D = response.json()
        r = D.get('results')
        listShips += [x.get(n) for x in r if x.get(n) and x.get(p) not in f]
        next = D.get('next')

    return listShips
