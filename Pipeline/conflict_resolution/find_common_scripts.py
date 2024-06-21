import requests, os, hashlib
import json

def find_common(fname, out_fname, empty_fname):

	hashdict = {}
	others = []
	empty = []
	errors = []
	ct = 0

	with open(fname) as f:
		setters = json.loads(f.read())['setters']
	print(len(setters))
	for setter in setters:
		try:
			print(ct)
			r = requests.get(setter, timeout=10)
			if (len(r.content)) > 0:
				if r.status_code == 200:
					hash_str = hashlib.sha1(r.content).hexdigest()
					if hash_str not in hashdict:
						hashdict[hash_str] = []
					hashdict[hash_str].append(setter)
				else:
					others.append((setter, r.status_code))
			else:
				empty.append(setter)
			
		except Exception as e:
			print(setter, e)
			errors.append(setter)
		ct += 1

		if ct % 100 == 0:
			with open(out_fname, "w") as f:
				f.write(json.dumps(hashdict, indent=4))
			with open(empty_fname, "w") as f:
				f.write(json.dumps({'empty' : empty, 'others' : others, 'errors' : errors}, indent=4))


if __name__ == "__main__":

	FNAME = "setters.json"
	OUT = "common.json"
	EMPTY = "empty.json"

	find_common(FNAME, OUT, EMPTY)