from openai import OpenAI


class DeepSeek:
    def __init__(self):
        # DeepSeek-V3.2 Tem=0.1  max_token=256
        self.client = OpenAI(api_key="api_key", base_url="https://api.deepseek.com")

    def request(self, api):

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": f"For {api} API, provide an introduction that includes its meaning, a sensitivity score from 0 to 5 based on network security concerns, potential malicious uses, and related APIs that might be used with it. The output should be in English, formatted as a single paragraph without line breaks or additional typography and as simple as possible."
        },
          ],
            max_tokens=256,
            temperature=0.1,
            stream=False,
            # seed=42
        )
        # NtProtectVirtualMemory
        """
        对NtProtectVirtualMemory这个API，输出一段文字评价，评价包括对API含义的解释，从0到5分以网络安全角度的敏感度进行评分，潜在的恶意用途，可能与之配合使用的其他API。输出内容为英文，格式为完整地一段话，不要有任何换行和分点等排版，尽可能简洁。
        For NtProtectVirtualMemory API, output a text evaluation, including an explanation of the meaning of the API, scoring from 0 to 5 on the sensitivity of the network security perspective, potential malicious use, and other apis that may be used with it. The output content is in English, the format is a complete paragraph, without any line breaks and other typography, as simple as possible.
        """

        return response.choices[0].message.content


if __name__ == '__main__':
    ds = DeepSeek()
    res = ds.request('NtProtectVirtualMemory')
    print(res)
