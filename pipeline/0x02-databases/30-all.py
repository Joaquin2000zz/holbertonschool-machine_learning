#!/usr/bin/env python3
"""
module which contains list_all function
"""


def list_all(mongo_collection):
    """
    lists all documents in a collection
    - mongo_collection will be the pymongo collection object
    Returns: an empty list if no document in the collection
    """
    return mongo_collection.find()
