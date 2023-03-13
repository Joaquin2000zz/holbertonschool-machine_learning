"""
module which contains sentientPlanets function
"""
import requests

def navigate(R, f, d, s, I=['unknown']):
    """
    given @R list, navigate into it's dictionaries

    """
    obtained = []
    for r in R:
        if r.get('designation') != d:
            continue
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
        if this and this not in I:
            obtained.append(this)

    return obtained
        
def sentientPlanets(designation='sentient', follow='homeworld', search='name'):
    """
    By using the Swapi API, creates a method that returns
    the list of names of the home planets of all
    species with sentient @designation from species route.
    """
    url = 'https://swapi-api.hbtn.io/api/species/'
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
    new = navigate(R=results, f=follow, d=designation, s=search)
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
        new = navigate(R=results, f=follow, d=designation, s=search)
        if not new:
            return listShips
        listShips += new

    return listShips
