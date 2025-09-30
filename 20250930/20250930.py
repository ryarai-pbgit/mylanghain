import getpass
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

if not os.environ.get("AZURE_OPENAI_API_KEY"):
  os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure: ")

from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

# ユーザーからの入力を取得
user_input = input("翻訳したい英語を入力してください: ")
target_language = input("翻訳先の言語を入力してください（例：Japanese, French, Spanish）: ")

# ChatPromptTemplateを使用
system_template = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

# プロンプトをフォーマット
formatted_prompt = prompt_template.format_messages(
    language=target_language,
    text=user_input
)

response = model.invoke(formatted_prompt)
print(response.content)