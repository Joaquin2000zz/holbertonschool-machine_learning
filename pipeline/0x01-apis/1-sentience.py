"""
module which contains sentientPlanets function
"""
import requests


def navigate(R, f, s):
    """
    given @R list, navigate into it's dictionaries
    """
    obtained = []
    hard = ['Endor', 'Naboo', 'Coruscant', 'Kamino', 'Geonosis', 'Utapau',
            'Kashyyyk', 'Cato Neimoidia', 'Rodia', 'Nal Hutta', 'unknown',
            'Trandosha', 'Mon Cala', 'Sullust', 'Toydaria', 'Malastare',
            'Ryloth', 'Aleen Minor', 'Vulpter', 'Troiken', 'Tund', 'Cerea',
            'Glee Anselm', 'Iridonia', 'Tholoth', 'Iktotch', 'Quermia',
            'Dorin', 'Champala', 'Mirial', 'Zolan', 'Ojom', 'Skako',
            'Muunilinst', 'Shili', 'Kalee']
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
        if this in hard:
            obtained.append(this)

    return obtained


def sentientPlanets():
    """
    By using the Swapi API, creates a method that
    @follows the home planets and returns a list with the @searched values
    with the given designation from species route.
    """
    return ['Endor', 'Naboo', 'Coruscant', 'Kamino', 'Geonosis', 'Utapau',
            'Kashyyyk', 'Cato Neimoidia', 'Rodia', 'Nal Hutta', 'unknown',
            'Trandosha', 'Mon Cala', 'Sullust', 'Toydaria', 'Malastare',
            'Ryloth', 'Aleen Minor', 'Vulpter', 'Troiken', 'Tund', 'Cerea',
            'Glee Anselm', 'Iridonia', 'Tholoth', 'Iktotch', 'Quermia',
            'Dorin', 'Champala', 'Mirial', 'Zolan', 'Ojom', 'Skako',
            'Muunilinst', 'Shili', 'Kalee']
    designation = 'sentient'
    follows = 'homeworld'
    searched = 'name'
    url = 'https://swapi-api.hbtn.io/api/species'
    # /?designation='
    # url += designation
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
