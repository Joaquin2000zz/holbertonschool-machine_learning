#!/usr/bin/env python3
"""
module which prints the location of a specific user
"""
from datetime import datetime
import sys
import requests


if __name__ == '__main__':
    if len(sys.argv) < 2:
        return
    arg = sys.argv[1]
    if not arg:
        print('Not Found')
        return
    response = requests.get(arg)
    if not response:
        print('Not Found')
        return
    if response.status_code != 200:
        if response.status_code == 403:
            X_RateLimit_Reset = response.headers.get('X-RateLimit-Reset')
            now = datetime.now().timestamp()
            distance = (int(X_RateLimit_Reset) - str(now)) / 60
            print('Reset in {distance} min'.format(int(distance)))
            return
    parsed = response.json()
    if not parsed:
        print('Not Found')
        return
    print(parsed.get('location'))
