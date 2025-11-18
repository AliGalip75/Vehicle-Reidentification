import subprocess
import json
import re
import os

# JSON'un backend/api içine kaydedileceği yol
OUTPUT_JSON = r"backend/api/metrics_static.json"

print("Running evaluate.py...")

process = subprocess.Popen(
    [
        r"C:\Users\galip\Desktop\vehicle_reid-master\.venv\Scripts\python.exe",
        "evaluate.py"
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

stdout, stderr = process.communicate()

if stderr.strip():
    print("Error:", stderr)

print("Raw output:")
print(stdout)

pattern = r"Rank@1:([\d\.]+)\s+Rank@5:([\d\.]+)\s+Rank@10:([\d\.]+)\s+mAP:([\d\.]+)"
match = re.search(pattern, stdout)

if not match:
    print("⚠ Çıktıda metrik bulunamadı!")
    exit()

rank1 = float(match.group(1))
rank5 = float(match.group(2))
rank10 = float(match.group(3))
mAP = float(match.group(4))

print("\nParsed metrics:")
print(rank1, rank5, rank10, mAP)

# JSON dosyasının tam yolu
save_path = os.path.join(os.path.dirname(__file__), OUTPUT_JSON)

# Klasör yoksa oluştur
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# JSON kaydet
with open(save_path, "w") as f:
    json.dump(
        {
            "rank1": rank1,
            "rank5": rank5,
            "rank10": rank10,
            "mAP": mAP
        },
        f,
        indent=4
    )

print(f"✔ Metrics saved to {save_path}")
