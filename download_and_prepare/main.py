from db_manager import DatabaseManager # type: ignore
from collections import Counter

id_counter = Counter()

for row in DatabaseManager.default().fetch_submissions():
  with open(f"data/human/{row[0]}_{id_counter[row[0]]}.json", "w") as sub_file:
    sub_file.write(row[1])
    id_counter.update({row[0]: 1})