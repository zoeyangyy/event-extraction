#!/usr/bin/env bash
#python wande-event.py --type=event --cuda=0
#python wande-test.py --type=event --cuda=0
#python wande-test.py --type=event --data=dev --cuda=2

#python wande-event.py --type=event --cf=gcn --cuda=2
#python wande-test.py --type=event --cf=gcn --cuda=2
#python wande-test.py --type=event --cf=gcn --data=dev --cuda=2

#python gcn-train.py --type=event --cf=gcn --cuda=2
python gcn-test.py --type=event --cf=gcn --cuda=2
python gcn-test.py --type=event --cf=gcn --cuda=2 --top=2
python gcn-test.py --type=event --cf=gcn --cuda=2 --top=3
