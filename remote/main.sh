#!/usr/bin/env bash
python wande-event.py --type=baseline
python wande-test.py --type=baseline
python wande-test.py --type=baseline --data=dev

python wande-event.py --type=baseline --cf=gcn
python wande-test.py --type=baseline --cf=gcn
python wande-test.py --type=baseline --cf=gcn --data=dev

python wande-event.py --type=time
python wande-test.py --type=time
python wande-test.py --type=time --data=dev

python wande-event.py --type=time --cf=gcn
python wande-test.py --type=time --cf=gcn
python wande-test.py --type=time --cf=gcn --data=dev

python wande-event.py --type=position
python wande-test.py --type=position
python wande-test.py --type=position --data=dev

python wande-event.py --type=position --cf=gcn
python wande-test.py --type=position --cf=gcn
python wande-test.py --type=position --cf=gcn --data=dev

python wande-event.py --type=event
python wande-test.py --type=event
python wande-test.py --type=event --data=dev

python wande-event.py --type=event --cf=gcn
python wande-test.py --type=event --cf=gcn
python wande-test.py --type=event --cf=gcn --data=dev

python wande-event.py --type=all
python wande-test.py --type=all
python wande-test.py --type=all --data=dev

python wande-event.py --type=all --cf=gcn
python wande-test.py --type=all --cf=gcn
python wande-test.py --type=all --cf=gcn --data=dev
