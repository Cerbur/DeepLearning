"""在项目目录使用 python -m corpus 下载、清洗并生成 train/eval。"""
from .download import main

if __name__ == "__main__":
    raise SystemExit(main())
