# LangChain + SerpAPI 问答示例

本项目演示如何使用 [LangChain](https://github.com/hwchase17/langchain) 结合 [SerpAPI](https://serpapi.com/) 进行问题解答。

## 安装依赖

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 环境变量设置

复制 `.env.example` 为 `.env` 文件：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填写你的 OpenAI 和 SerpAPI 的 API 密钥：

```text
OPENAI_API_KEY=你的openai密钥
SERPAPI_API_KEY=你的serpapi密钥
```

你也可以直接在命令行中导出环境变量：

```bash
export OPENAI_API_KEY=你的openai密钥
export SERPAPI_API_KEY=你的serpapi密钥
```

## 使用方法

```bash
python main.py 你要提问的问题
```

例如：

```bash
python main.py "今天上海的天气怎么样？"
```

## 说明

- 代码会自动调用 SerpAPI 进行搜索，并用 OpenAI 进行答案生成。
- 需要科学上网以访问 OpenAI 和 SerpAPI。
