#!/usr/bin/env python3
"""
module which contains insert_school function
"""


def insert_school(mongo_collection, **kwargs):
    """
    inserts a new document in a collection based on kwargs:
    - mongo_collection will be the pymongo collection object
    Returns: the new _id
    """
    return mongo_collection.school.insert_one(kwargs)
