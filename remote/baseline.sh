#!/usr/bin/env bash
python wande-event.py --type=baseline --cuda=3
python wande-test.py --type=baseline --cuda=3

#python wande-test.py --type=baseline --data=dev --cuda=1

#python wande-event.py --type=baseline --cf=gcn --cuda=0
#python wande-test.py --type=baseline --cf=gcn --cuda=0
#python wande-test.py --type=baseline --cf=gcn --data=dev --cuda=1

#python gcn-train.py --type=baseline --cf=gcn --cuda=0
#python gcn-test.py --type=baseline --cf=gcn --cuda=0
