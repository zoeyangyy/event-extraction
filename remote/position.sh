#!/usr/bin/env bash
#python wande-event.py --type=position --cuda=3
#python wande-test.py --type=position --cuda=3
#python wande-test.py --type=position --data=dev --cuda=1

#python wande-event.py --type=position --cf=gcn --cuda=1
#python wande-test.py --type=position --cf=gcn --cuda=1
#python wande-test.py --type=position --cf=gcn --data=dev --cuda=1

#python gcn-train.py --type=position --cf=gcn --cuda=3
python gcn-test.py --type=position --cf=gcn --cuda=3
python gcn-test.py --type=position --cf=gcn --cuda=3 --top=2
python gcn-test.py --type=position --cf=gcn --cuda=3 --top=3