"""语料下载、字符清洗和离线整理；仅依赖 Python 标准库。"""
from .cleaning import clean_corpus_text

__all__ = ["clean_corpus_text"]
