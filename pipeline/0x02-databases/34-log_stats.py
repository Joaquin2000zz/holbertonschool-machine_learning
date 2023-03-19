#!/usr/bin/env python3
"""
script that provides some stats about Nginx logs stored in MongoDB:

- Database: logs
- Collection: nginx
- Display (same as the example):
- first line: x logs where x is the number of documents in this collection
- second line: Methods:
- 5 lines with the n° of documents with the method = ["GET", "POST", "PUT",
  "PATCH", "DELETE"] in this order (see example below - warning:
  it’s a tabulation before each line)
- one line with the number of documents with:
- method=GET
- path=/status
"""
from pymongo import MongoClient


if __name__ == '__main__':
    client = MongoClient('mongodb://127.0.0.1:27017')
    collection = client.logs.nginx
    boundaries = ["DELETE", "GET", "PATCH", "POST", "PUT"]
    aggregation = {'$bucket': {'groupBy': '$method',
                               'boundaries': boundaries,
                               'output': {'total': {'$sum': 1}}}}
    methods = collection.aggregate([aggregation])
    methods = {x.get('_id'): x.get('total') for x in methods}
    boundaries = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    print(collection.count_documents({}), 'logs')
    print("Methods:")
    for b in boundaries:
        print('\tmethod {}:'.format(b),
              methods.get(b) if methods.get(b) else 0)
    check = collection.aggregate([{'$match': {'path': '/status'}},
                                  {'$count': "check"}])
    check = list(check._CommandCursor__data)[0].get('check')
    print(check, 'status check')
