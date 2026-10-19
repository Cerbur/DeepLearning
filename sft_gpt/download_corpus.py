import io
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

# Ensure stdout is utf-8
sys.stdout.reconfigure(encoding="utf-8")

URL = "https://hf-mirror.com/datasets/pleisto/wikipedia-cn-20230720-filtered/resolve/main/wikipedia-cn-20230720-filtered.json"
TARGET_DIR = Path(".data/corpus")
TARGET_FILE = TARGET_DIR / "wiki_zh_pretrain.jsonl"
TARGET_DOCS = 50000  # 50,000 high-quality Chinese articles
MIN_DOC_LEN = 50     # Filter out stubs / navigation fragments

TARGET_DIR.mkdir(parents=True, exist_ok=True)

print(f"=== 准备下载并处理高质量中文预训练语料 (Wikipedia-CN) ===")
print(f"数据源: {URL}")
print(f"目标路径: {TARGET_FILE.resolve()}")
print(f"目标文档数: {TARGET_DOCS} 条")

start_time = time.time()
req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})

decoder = json.JSONDecoder()
buffer = ""
saved_count = 0
total_chars = 0
chunk_size = 1024 * 1024  # 1MB buffer reads

with urllib.request.urlopen(req, timeout=60) as resp, TARGET_FILE.open("w", encoding="utf-8") as out_f:
    print("已连接到数据源，正在流式读取与解析...")
    downloaded_bytes = 0
    last_report_time = time.time()
    
    while saved_count < TARGET_DOCS:
        chunk = resp.read(chunk_size)
        if not chunk:
            print("数据源读取完毕。")
            break
            
        downloaded_bytes += len(chunk)
        buffer += chunk.decode("utf-8", errors="ignore")
        
        # Strip leading brackets or commas if at start
        pos = 0
        while pos < len(buffer) and buffer[pos] in " \r\n\t,[":
            pos += 1
            
        while pos < len(buffer) and saved_count < TARGET_DOCS:
            while pos < len(buffer) and buffer[pos] in " \r\n\t,":
                pos += 1
            if pos >= len(buffer) or buffer[pos] == "]":
                break
            try:
                obj, end_pos = decoder.raw_decode(buffer, pos)
                pos = end_pos
                
                # Extract text
                text = obj.get("completion", "")
                if text:
                    # Clean up: compress excessive blank lines and trailing spaces
                    text = re.sub(r"\n{3,}", "\n\n", text).strip()
                    if len(text) >= MIN_DOC_LEN:
                        record = {"text": text}
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        saved_count += 1
                        total_chars += len(text)
            except json.JSONDecodeError:
                # Incomplete JSON object at end of buffer; retain remaining buffer for next chunk
                break
                
        # Keep remaining unparsed buffer
        buffer = buffer[pos:]
        
        # Periodic progress report every 5 seconds
        now = time.time()
        if now - last_report_time >= 5 or saved_count >= TARGET_DOCS:
            elapsed = now - start_time
            speed = (downloaded_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
            mb = downloaded_bytes / (1024 * 1024)
            print(f"进度: 已处理 {saved_count}/{TARGET_DOCS} 条文档 | 下载: {mb:.1f} MB ({speed:.2f} MB/s) | 字符数: {total_chars:,}")
            last_report_time = now

elapsed = time.time() - start_time
file_size_mb = TARGET_FILE.stat().st_size / (1024 * 1024)
print(f"\n=== 完成处理与落地 ===")
print(f"生成文件: {TARGET_FILE}")
print(f"文件大小: {file_size_mb:.2f} MB")
print(f"文档总数: {saved_count:,} 条")
print(f"汉字/字符总数: {total_chars:,} 字符 (预计约 {int(total_chars * 1.2):,} tokens)")
print(f"总耗时: {elapsed:.1f} 秒")
