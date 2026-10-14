"""Rebuild tests/fixtures/own-data/answers.csv: 72 synthetic open-ended question-answer rows (invented text, no license).

    python tests/fixtures/own-data/make_answers.py
"""
import csv, random
rng = random.Random(7)
things = ["lighthouse", "bridge", "library", "observatory", "canal", "market", "tower", "harbor", "museum", "railway"]
places = ["Northport", "Eastvale", "Southbrook", "Westfield", "Lakeside", "Ridgemont"]
years = list(range(1820, 1990, 7))
facts = [
    ("was completed in {y}", "When was the {t} in {p} completed?", "It was completed in {y}."),
    ("was designed by {a}", "Who designed the {t} in {p}?", "{a} designed it."),
    ("is open to visitors on weekends", "When can visitors enter the {t} in {p}?", "On weekends."),
    ("was restored after a fire in {y}", "What happened to the {t} in {p} in {y}?", "It was restored after a fire."),
]
architects = ["Maren Holt", "Idris Vale", "Corin Ashby", "Lena Quist", "Tobias Rennick"]
rows = []
for i in range(72):
    t, p, y, a = things[i % len(things)], places[(i // 2) % len(places)], years[i % len(years)] + (i // len(years)), architects[i % len(architects)]
    fact, q, ans = facts[i % len(facts)]
    passage = f"The {t} in {p} {fact.format(y=y, a=a)}. It is one of several landmarks in the region and is visited by schools each spring (entry {i})."
    rows.append((f"Passage: {passage} Question: {q.format(t=t, p=p, y=y, a=a)}", ans.format(y=y, a=a)))
with open("tests/fixtures/own-data/answers.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["prompt", "answer"])
    w.writerows(rows)
print(len(rows), len({r[0] for r in rows}))
