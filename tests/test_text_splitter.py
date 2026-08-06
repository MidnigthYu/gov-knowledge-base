import unittest
from utils.text_splitter import TextSplitter

class TestTextSplitter(unittest.TestCase):

    def test_normal_split(self):
        text = "a" * 1000
        splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split(text)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertLessEqual(len(chunks[0]["content"]), 500)

    def test_overlap_exist(self):
        text = "第一段内容。" * 100
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = splitter.split(text)
        if len(chunks) >= 2:
            # 块1末尾与块2开头存在重叠
            self.assertTrue(chunks[1]["content"].startswith(chunks[0]["content"][-20:]))

    def test_short_text(self):
        text = "短文本测试"
        splitter = TextSplitter(chunk_size=500)
        chunks = splitter.split(text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["content"], text)

    def test_empty_none_input(self):
        splitter = TextSplitter()
        self.assertEqual(splitter.split(""), [])
        self.assertEqual(splitter.split(None), [])
        self.assertEqual(splitter.split(123), [])

    def test_meta_info_correct(self):
        text = "第一段落\n\n第二段落\n\n第三段落"
        splitter = TextSplitter(chunk_size=20, chunk_overlap=0)
        chunks = splitter.split(text)
        for chunk in chunks:
            self.assertIn("chunk_index", chunk)
            self.assertIn("start_pos", chunk)
            self.assertEqual(text[chunk["start_pos"]:chunk["start_pos"]+5], chunk["content"][:5])


if __name__ == "__main__":
    unittest.main()