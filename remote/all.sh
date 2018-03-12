#!/usr/bin/env bash
#python wande-event.py --type=all --cuda=3
#python wande-test.py --type=all --cuda=3
python wande-test.py --type=all --data=dev --cuda=3

#python wande-event.py --type=all --cf=gcn --cuda=3
#python wande-test.py --type=all --cf=gcn --cuda=3
python wande-test.py --type=all --cf=gcn --data=dev --cuda=3