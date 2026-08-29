import unittest
import os
from app.utils.document_parser import DocumentParser

class TestDocumentParser(unittest.TestCase):
    def setUp(self):
        self.test_files_dir = "tests/test_files"

    def test_parse_valid_txt_file(self):
        """合法TXT文件正常解析，返回清洗后的纯文本字符串"""
        test_file = os.path.join(self.test_files_dir, "utf8_test.txt")
        result = DocumentParser.parse(test_file)
        
        self.assertIsInstance(result, str)
        self.assertGreater(len(result.strip()), 0)

    def test_parse_unsupported_format(self):
        """不支持的文件格式（扩展名拦截）抛出 ValueError"""
        invalid_file = os.path.join(self.test_files_dir, "invalid_sample.exe")
        with self.assertRaises(ValueError):
            DocumentParser.parse(invalid_file)

    def test_batch_parse_directory(self):
        """批量解析目录，返回(路径, 文本)元组列表，有效文件数量大于0"""
        results = DocumentParser.batch_parse(self.test_files_dir)
        
        self.assertIsInstance(results, list)
        self.assertIsInstance(results[0], tuple)
        self.assertEqual(len(results[0]), 2)
        self.assertGreater(len(results), 0)
