#!/usr/bin/env bash
#python wande-event.py --type=position --cuda=1
#python wande-test.py --type=position --cuda=1
python wande-test.py --type=position --data=dev --cuda=1

#python wande-event.py --type=position --cf=gcn --cuda=1
#python wande-test.py --type=position --cf=gcn --cuda=1
python wande-test.py --type=position --cf=gcn --data=dev --cuda=1
