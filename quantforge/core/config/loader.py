import json

def load(path):

    with open(path) as fp:

        return json.load(fp)
