import json

with open('data/pretraining_data.json', encoding='utf-8') as f:
    data = json.load(f)

result = []
for s in data:
    s = s.rstrip(' </s>')
    words = s.split()
    if len(words) != 3:
        result.append(s + ' </s>')

print(json.dumps(result))