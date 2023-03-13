#!/usr/bin/env python3
"""
module which prints the location of a specific user
"""
import json as js
import sys
import requests


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Not Found')
        exit()
    arg = sys.argv[1]
    if not arg:
        print('Not Found')
        exit()
    response = requests.get(arg)
    if not response:
        print('Not Found')
        exit()
    if response.status_code != 200:
        if response.status_code == 403:
            print('Reset in X min')
            exit()
    parsed = js.loads(response.text)
    if not parsed:
        print('Not Found')
        exit()
    print(parsed.get('location'))
