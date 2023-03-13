"""
module which contains sentientPlanets function
"""
import requests


def navigate(R, f, s, ignore=['unknown']):
    """
    given @R list, navigate into it's dictionaries

    """
    obtained = []
    for r in R:
        this = r.get(f)
        if not this:
            continue
        response = requests.get(this)
        if not isinstance(response, requests.Response):
            return []
        if response.status_code != 200:
            return []

        json_response = response.json()
        if not json_response:
            return []
        this = json_response.get(s)
        if this and this not in ignore:
            obtained.append(this)

    return obtained


def sentientPlanets():
    """
    By using the Swapi API, creates a method that
    @follows the home planets and returns a list with the @searched values
    with the given designation from species route.
    """
    designation='sentient'
    follows='homeworld'
    searched='name'
    url = 'https://swapi-api.hbtn.io/api/species/?designation='
    url += designation
    response = requests.get(url)
    if not isinstance(response, requests.Response):
        return []
    if response.status_code != 200:
        return []

    json_response = response.json()
    if not json_response:
        return []

    results = json_response.get('results')
    if not results:
        return []
    new = navigate(R=results, f=follows, s=searched)
    if not new:
        return []
    listShips = new
    json_response = response.json()
    next = json_response.get('next')

    while response and isinstance(listShips, list) and next:
        response = requests.get(next)
        json_response = response.json()
        next = json_response.get('next')

        if not json_response:
            return listShips
        results = json_response.get('results')
        if not results:
            return listShips
        new = navigate(R=results, f=follows, s=searched)
        if not new:
            return listShips
        listShips += new

    return listShips
