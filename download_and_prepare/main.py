from db_manager import DatabaseManager # type: ignore
from collections import Counter

id_counter = Counter()

round_label = 2

invite_groups = [
  "Mrs. Ashley Email - Round 3"
]

submissions = []

for group in invite_groups:
  submissions += DatabaseManager.default().fetch_submissions_from_group(group)

for row in submissions:
  with open(f"data/human/{row[0]}_{id_counter[row[0]]}_round{round_label}.json", "w") as sub_file:
    sub_file.write(row[1])
    id_counter.update({row[0]: 1})