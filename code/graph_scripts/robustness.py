import pandas as pd
import random
import json
import string
from tqdm.auto import tqdm
import tldextract


def generate_random_string(length=5):
    return ''.join(random.choice(string.ascii_lowercase + string.ascii_uppercase + string.digits) for i in range(length))


def generate_random_domain(length=5):
    tlds = ['com', 'net', 'org', 'edu', 'gov', 'io']
    return ''.join(generate_random_string(length) + '.' + random.choice(tlds))