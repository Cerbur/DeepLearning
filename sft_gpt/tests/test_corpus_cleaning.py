"""中文/英文/ASCII数字/标点白名单的独立标准库测试。"""
import string
import unittest

try:
    from sft_gpt.corpus.cleaning import clean_corpus_text
except ModuleNotFoundError:
    from corpus.cleaning import clean_corpus_text


class CorpusCleaningTests(unittest.TestCase):
    def test_allowed_text_and_whitespace_are_preserved(self):
        text = "  中文繁體𠮷〇 ABCxyz 0123456789，。！？《书》【引号】…—\t\r\n" + string.punctuation
        self.assertEqual(clean_corpus_text(text), text)

    def test_foreign_scripts_emoji_accents_and_non_ascii_numbers_are_removed(self):
        foreign = "あいうカナ한글ПриветΕλληνικάالعربيةहिन्दीéñ１２３١٢٣😊🦊☀️"
        self.assertEqual(clean_corpus_text("中文" + foreign + "English 123。"),
                         "中文 English 123。")
        self.assertEqual(clean_corpus_text(foreign), "")

    def test_rejected_runs_do_not_join_neighboring_words(self):
        self.assertEqual(clean_corpus_text("hello🙂мирworld"), "hello world")
        self.assertEqual(clean_corpus_text("中文🙂 English\n俄语Привет再见"),
                         "中文  English\n俄语 再见")

    def test_punctuation_and_whitespace_only_are_empty(self):
        for text in ("", " \t\r\n", "，。!?", string.punctuation, "😊!?あ"):
            with self.subTest(text=text):
                self.assertEqual(clean_corpus_text(text), "")

    def test_cleanup_is_idempotent(self):
        for text in ("abc俄语Привет🦊中文！", "English\t\r\n中文。", "あいう"):
            cleaned = clean_corpus_text(text)
            self.assertEqual(clean_corpus_text(cleaned), cleaned)

    def test_invalid_input_type_is_explicit(self):
        for value in (None, 12, [], {}):
            with self.subTest(value=value), self.assertRaises(TypeError):
                clean_corpus_text(value)
