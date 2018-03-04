#!/usr/bin/env bash
python wande-event.py --type=position
python wande-test.py --type=position
python wande-event.py --type=event
python wande-test.py --type=event
