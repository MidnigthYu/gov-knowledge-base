"""
文本清洗工具单元测试
覆盖空行去除、特殊字符过滤、空白规范化、编码检测读取与批量清洗等场景
依赖：unittest、text_cleaner
"""
import unittest
import os
import tempfile
from app.utils.text_cleaner import clean_single_text, safe_read_file, batch_clean_files


class TestCleanSingleText(unittest.TestCase):
    """基础文本清洗功能测试"""

    def test_remove_empty_lines(self):
        text = "第一行\n\n\n第二行\n\n第三行"
        result = clean_single_text(text)
        self.assertEqual(result, "第一行\n第二行\n第三行")

    def test_remove_special_chars(self):
        text = "正常文本\t\r\x0b\x0c特殊字符"
        result = clean_single_text(text)
        self.assertNotIn("\t", result)
        self.assertNotIn("\r", result)

    def test_whitespace_normalize(self):
        text = "  多个   空格  测试  "
        result = clean_single_text(text)
        self.assertEqual(result, "多个 空格 测试")

    def test_empty_input(self):
        self.assertEqual(clean_single_text(""), "")
        self.assertEqual(clean_single_text(None), "")


class TestSafeReadFile(unittest.TestCase):
    """文件安全读取功能测试"""
    TEST_DIR = os.path.join(os.path.dirname(__file__), "test_files")

    def test_utf8_file_read(self):
        path = os.path.join(self.TEST_DIR, "utf8_test.txt")
        content = safe_read_file(path)
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 0)

    def test_gbk_file_auto_detect(self):
        path = os.path.join(self.TEST_DIR, "gbk_test.txt")
        content = safe_read_file(path)
        self.assertIsInstance(content, str)
        self.assertNotIn("锟斤拷", content)

    def test_empty_file_return_empty(self):
        path = os.path.join(self.TEST_DIR, "empty_file.txt")
        content = safe_read_file(path)
        self.assertEqual(content, "")

    def test_corrupt_file_no_crash(self):
        path = os.path.join(self.TEST_DIR, "corrupt_file.txt")
        # 损坏文件不抛出异常，返回空或可读内容
        try:
            content = safe_read_file(path)
            self.assertTrue(content is None or isinstance(content, str))
        except Exception as e:
            self.fail(f"读取损坏文件抛出异常: {e}")


class TestBatchCleanFiles(unittest.TestCase):
    """批量清洗功能测试"""

    def test_recursive_scan_and_structure_preserve(self):
        """测试递归扫描与目录结构保持"""
        input_dir = os.path.join(os.path.dirname(__file__), "test_files")
        with tempfile.TemporaryDirectory() as output_dir:
            success, failed = batch_clean_files(input_dir, output_dir)
            self.assertGreaterEqual(success, 3)
            # 验证子目录文件也被处理
            sub_out = os.path.join(output_dir, "sub_dir")
            self.assertTrue(os.path.exists(sub_out))

    def test_exception_isolation(self):
        """损坏文件不影响整体任务"""
        input_dir = os.path.join(os.path.dirname(__file__), "test_files")
        with tempfile.TemporaryDirectory() as output_dir:
            success, failed = batch_clean_files(input_dir, output_dir)
            self.assertGreater(success, 0)
            self.assertGreaterEqual(failed, 0)


if __name__ == "__main__":
    unittest.main()