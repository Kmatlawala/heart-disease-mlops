"""Send sample traffic to the API (to test it and to fill the Grafana dashboard).

Usage:
  python scripts/send_requests.py                          # localhost:8000, 50 requests
  python scripts/send_requests.py --url http://localhost:80 --n 200
"""
import argparse
import json
import random
import time
import urllib.error
import urllib.request

BASE = json.load(open(__file__.replace("send_requests.py", "sample_request.json")))


def random_patient():
    p = dict(BASE)
    p.update(age=random.randint(29, 77), sex=random.randint(0, 1), cp=random.randint(1, 4),
             trestbps=random.randint(94, 200), chol=random.randint(126, 564),
             thalach=random.randint(71, 202), exang=random.randint(0, 1),
             oldpeak=round(random.uniform(0, 6.2), 1), ca=random.randint(0, 3),
             thal=random.choice([3, 6, 7]), slope=random.randint(1, 3))
    return p


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    print("health:", urllib.request.urlopen(args.url + "/health").read().decode())
    print("sample:", post(args.url + "/predict", BASE))
    for i in range(args.n):
        body = random_patient()
        if i % 15 == 14:          # a few bad requests so error metrics show up
            body["age"] = 500
        status, out = post(args.url + "/predict", body)
        print(f"{i + 1:>3} {status} {out.get('label', out)} {out.get('confidence', '')}")
        time.sleep(0.1)
