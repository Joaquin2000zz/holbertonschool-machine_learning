#!/usr/bin/env python3
"""
module which prints the location of a specific user
"""
from datetime import datetime
import sys
import requests


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Not Found')
    else:
        arg = sys.argv[1]
        if not arg:
            print('Not Found')

        response = requests.get(arg)
        if response.status_code == 404:
            print('Not Found')
        if response.status_code == 200:
            parsed = response.json()
            message = parsed.get('message')
            if message == 'Not Found':
                print(message)
            else:
                print(
                    parsed.get('location')
                )
        if response.status_code == 403:
            X_RateLimit_Reset = response.headers.get(
                'X-RateLimit-Reset'
            )
            now = datetime.now().timestamp()
            distance = (int(X_RateLimit_Reset) - int(now)) / 60
            print('Reset in {} min'.format(int(distance)))
