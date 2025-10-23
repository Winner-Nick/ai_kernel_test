"""
自定义推理函数 - 使用我们的 API 中转，但返回 KernelBench 兼容的格式
"""
import time
from openai import OpenAI
from .config import API_BASE_URL, API_KEY


def create_custom_inference_function(model_id, temperature=0.0, max_tokens=4096, verbose=False):
    """
    创建一个自定义推理函数，兼容 KernelBench 的接口

    Args:
        model_id: 模型 ID
        temperature: 温度参数
        max_tokens: 最大 token 数
        verbose: 是否打印详细信息

    Returns:
        inference_fn: 推理函数，接受 prompt 返回生成的文本
    """
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    def inference_fn(prompt: str) -> str:
        """
        推理函数

        Args:
            prompt: 输入提示词

        Returns:
            str: 生成的文本
        """
        if verbose:
            print(f"🤖 调用模型: {model_id}")
            print(f"📝 Prompt 长度: {len(prompt)} 字符")

        start_time = time.time()

        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert GPU programmer specializing in CUDA kernel optimization."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            generated_text = response.choices[0].message.content
            elapsed_time = time.time() - start_time

            if verbose:
                print(f"✅ 生成成功! 耗时: {elapsed_time:.2f}s")
                print(f"📊 生成长度: {len(generated_text)} 字符")

            return generated_text

        except Exception as e:
            if verbose:
                print(f"❌ 生成失败: {str(e)}")
            raise e

    return inference_fn
