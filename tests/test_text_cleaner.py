import unittest
from utils.text_cleaner import clean_single_text

class TestTextCleaner(unittest.TestCase):
    """文本清洗工具单元测试"""

    def test_normal_text(self):
        """测试正常文本清洗"""
        raw = "你好   世界"
        result = clean_single_text(raw)
        self.assertEqual(result, "你好 世界")

    def test_empty_lines(self):
        """测试空行过滤"""
        raw = "第一行\n\n\n第二行"
        result = clean_single_text(raw)
        self.assertEqual(result, "第一行\n第二行")

    def test_special_chars(self):
        """测试特殊字符过滤"""
        raw = "测试★●◆特殊符号"
        result = clean_single_text(raw)
        self.assertEqual(result, "测试特殊符号")

    def test_empty_input(self):
        """测试空文本边界场景"""
        raw = ""
        result = clean_single_text(raw)
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()