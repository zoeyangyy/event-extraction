import json
import collections
import datetime


with open('/Users/zoe/Documents/event_extraction/majorEventDump/majorEventDump.json', 'r', encoding='utf-8') as inputFile:
    events = json.load(inputFile)
with open('/Users/zoe/Documents/event_extraction/majorEventDump/typeCodeDump.json', 'r', encoding='utf-8') as inputFile:
    code2type = json.load(inputFile)

TIME_WINDOW = datetime.timedelta(30)

eventsGroupByCompany = collections.defaultdict(list)
for event in events:
    try:
        company = event['S_INFO_WINDCODE']
        parseEvent = {
            'type': code2type[event['S_EVENT_CATEGORYCODE']],
            'date': datetime.datetime.strptime(event['S_EVENT_HAPDATE'], '%Y%m%d'),
        }
        eventsGroupByCompany[company].append(parseEvent)

    except:
        continue


freqDist = collections.defaultdict(int)
pairFreqDist = collections.defaultdict(int)
trigramFreqDist = collections.defaultdict(int)

for company, eventSeq in list(eventsGroupByCompany.items())[0:5]:
    print(company)

    sortedEventSeq = sorted(eventSeq, key=lambda e: e['date'])
    for beginIdx, e in enumerate(sortedEventSeq):
        freqDist[e['type']] += 1

        date = e['date']
        for idx in range(beginIdx + 1, len(sortedEventSeq)):
            e_aft = sortedEventSeq[idx]
            if e_aft['date'] - date > TIME_WINDOW:
                break
            else:
                pairFreqDist[(e['type'], e_aft['type'])] += 1

            for idx2 in range(idx + 1, len(sortedEventSeq)):
                e_aft2 = sortedEventSeq[idx2]
                if e_aft2['date'] - date > TIME_WINDOW:
                    break
                else:
                    trigramFreqDist[(e['type'], e_aft['type'], e_aft2['type'])] += 1

pairRank = collections.defaultdict(float)

for pair, freq in pairFreqDist.items():
    w = freq / freqDist[pair[0]] / freqDist[pair[1]]
    pairRank[tuple(sorted(pair))] += w

trigramRank = collections.defaultdict(float)
for trigram, freq in trigramFreqDist.items():
    w = freq / freqDist[trigram[0]] / freqDist[trigram[1]] / freqDist[trigram[2]]
    trigramRank[tuple(sorted(trigram))] += w


# with open('../output/freqDist.txt', 'w', encoding='utf-8') as outputFile:
#     for unigram, freq in sorted(freqDist.items(), key=lambda pf: pf[1], reverse=True):
#         outputFile.write(unigram + '\t' + str(freq) + '\n')
#
# with open('../output/pairFreqDist.txt', 'w', encoding='utf-8') as outputFile:
#     for pair, freq in sorted(pairFreqDist.items(), key=lambda pf: pf[1], reverse=True):
#         outputFile.write('\t'.join(pair) + '\t' + str(freq) + '\n')
#
# with open('../output/pairRank.txt', 'w', encoding='utf-8') as outputFile:
#     for pair, w in sorted(pairRank.items(), key=lambda pf: pf[1], reverse=True):
#         outputFile.write('\t'.join(pair) + '\t' + str(w) + '\n')
#
# with open('../output/trigramFreqDist.txt', 'w', encoding='utf-8') as outputFile:
#     for triple, freq in sorted(trigramFreqDist.items(), key=lambda pf: pf[1], reverse=True):
#         outputFile.write('\t'.join(triple) + '\t' + str(freq) + '\n')
#
# with open('../output/trigramRank.txt', 'w', encoding='utf-8') as outputFile:
#     for triple, w in sorted(trigramRank.items(), key=lambda pf: pf[1], reverse=True):
#         outputFile.write('\t'.join(triple) + '\t' + str(w) + '\n')
