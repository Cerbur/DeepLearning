
import os, glob
from opencc import OpenCC
cc = OpenCC('t2s')

for root, dirs, files in os.walk("temp_cp"):
    for file in files:
        if file.endswith(".json"):
            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if "生當作人傑" in content or "生当作人杰" in cc.convert(content):
                        print("Found in:", path)
                    elif "不肯過江東" in content:
                        print("Found by jiangdong:", path)
            except Exception:
                pass
