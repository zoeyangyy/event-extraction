#!/usr/bin/env bash
python wande-event.py --type=baseline --cf=gcn
python wande-test.py --type=baseline --cf=gcn
