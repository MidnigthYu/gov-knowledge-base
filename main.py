from common.logger import get_logger
from client.zhipu_client import ZhipuClient
from client.deepseek_client import DeepSeekClient
from utils.text_cleaner import batch_clean_files
from pathlib import Path

logger = get_logger("main")

def test_llm_clients():
    """测试两款大模型的单轮调用"""
    test_prompt = "你好，请用一句话介绍一下你自己。"

    logger.info("=== 测试智谱AI ===")
    zhipu = ZhipuClient()
    zhipu_reply = zhipu.chat(test_prompt)
    logger.info(f"智谱回复：{zhipu_reply}")

    logger.info("=== 测试DeepSeek ===")
    deepseek = DeepSeekClient()
    deepseek_reply = deepseek.chat(test_prompt)
    logger.info(f"DeepSeek回复：{deepseek_reply}")

def test_multi_round():
    client = ZhipuClient()
    print("=== 第一轮 ===")
    print(client.chat("我叫小明"))

    print("\n=== 第二轮 ===")
    print(client.chat("我叫什么名字？"))

    print("\n=== 清空历史，开新会话 ===")
    client.clear_history()
    print(client.chat("我叫什么名字？"))
    
def test_text_cleaner():
    """测试文本批量清洗工具"""
    logger.info("=== 测试文本清洗 ===")
    script_dir = Path(__file__).resolve().parent
    input_dir = script_dir / "data" / "input"
    output_dir = script_dir / "data" / "output"
    
    batch_clean_files(input_dir, output_dir)

if __name__ == "__main__":
    test_llm_clients()
    test_text_cleaner()
    test_multi_round()