#!/usr/bin/env bash
#python wande-event.py --type=time --cuda=2
#python wande-test.py --type=time --cuda=2
#python wande-test.py --type=time --data=dev --cuda=2

#python wande-event.py --type=time --cf=gcn --cuda=2
#python wande-test.py --type=time --cf=gcn --cuda=2
#python wande-test.py --type=time --cf=gcn --data=dev --cuda=2

#python gcn-train.py --type=time --cf=gcn --cuda=2
python gcn-test.py --type=time --cf=gcn --cuda=2
python gcn-test.py --type=time --cf=gcn --cuda=2 --top=2
python gcn-test.py --type=time --cf=gcn --cuda=2 --top=3