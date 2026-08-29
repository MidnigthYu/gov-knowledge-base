"""
文本清洗工具单元测试
覆盖空行去除、特殊字符过滤、空白规范化、编码检测读取与批量清洗等场景
依赖：unittest、text_cleaner
"""
import unittest
import os
import tempfile
from app.utils.text_cleaner import clean_single_text, safe_read_file, batch_clean_files, clean_government_text


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


class TestGovernmentTextClean(unittest.TestCase):
    """政务公文清洗专项测试"""

    def setUp(self):
        """标准红头公文测试样例：包含文号、红头横线、页码、单位+日期落款"""
        self.standard_gov_doc = """武汉东湖新技术开发区管理委员会文件

武新管〔2024〕8号

───────────────────────

东湖高新区3551人才举荐政策实施办法

第一章 总则
为深入实施人才强区战略，加快集聚高层次创新创业人才，制定本办法。

第 2 页 共 3 页

第二章 举荐对象与条件
面向全球范围内的高层次创新创业人才，重点支持光电、生物医药等产业。

武汉东湖新技术开发区管理委员会
2024年2月20日
"""
        # 普通文档：正文包含单位名和日期，验证不误删
        self.normal_doc = """普通业务说明文档
第一条 适用范围
本规则由武汉市财政局负责解释，自2024年3月15日起施行。
第二条 其他事项
未尽事宜另行通知。"""

    def test_remove_document_number(self):
        """测试发文字号被完整删除"""
        result = clean_government_text(self.standard_gov_doc)
        self.assertNotIn("武新管〔2024〕8号", result)

    def test_remove_red_line_separator(self):
        """测试红头分隔横线被完整删除"""
        result = clean_government_text(self.standard_gov_doc)
        self.assertNotIn("───────────────────────", result)

    def test_remove_page_number(self):
        """测试页码行被完整删除"""
        result = clean_government_text(self.standard_gov_doc)
        self.assertNotIn("第 2 页 共 3 页", result)

    def test_remove_signature_pair(self):
        """测试单位+日期落款成对删除"""
        result = clean_government_text(self.standard_gov_doc)
        self.assertNotIn("武汉东湖新技术开发区管理委员会\n2024年2月20日", result)

    def test_keep_body_content(self):
        """测试正文核心内容完整保留，无误删"""
        result = clean_government_text(self.standard_gov_doc)
        self.assertIn("第一章 总则", result)
        self.assertIn("面向全球范围内的高层次创新创业人才", result)

    def test_normal_document_no_side_effect(self):
        """测试普通文档无副作用：正文内的单位、日期不被误删"""
        result = clean_government_text(self.normal_doc)
        self.assertIn("武汉市财政局", result)
        self.assertIn("2024年3月15日", result)

    def test_remove_prefix_wenhao(self):
        """测试带前缀的文号行被完整删除"""
        text = "文号：武财税〔2024〕12号\n第一章 总则\n政策内容"
        result = clean_government_text(text)
        self.assertNotIn("文号：", result)
        self.assertIn("第一章 总则", result)

    def test_remove_red_header_file(self):
        """测试红头文件抬头行被完整删除"""
        text = "武汉市财政局文件\n第一章 总则\n政策内容"
        result = clean_government_text(text)
        self.assertNotIn("武汉市财政局文件", result)
        self.assertIn("第一章 总则", result)

    def test_keep_inline_wenhao(self):
        """测试正文内嵌的引用文号完整保留，不误删"""
        text = "根据《高新技术企业认定管理办法》（国科发火〔2016〕32号）执行"
        result = clean_government_text(text)
        self.assertIn("国科发火〔2016〕32号", result)
        self.assertIn("根据《高新技术企业认定管理办法》", result)

if __name__ == "__main__":
    unittest.main()