# 実行中しか会話履歴が残らないスクリプト

from langchain_core.messages import SystemMessage

from boto3.session import Session
from langchain.chat_models import BedrockChat
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

import os
from dotenv import load_dotenv

load_dotenv()

session = Session(profile_name=os.environ["AWS_PROFILE"])
region = os.environ["AWS_REGION"]
model_id = os.environ["MODEL_ID"]

bedrock_runtime = session.client("bedrock-runtime", region_name=region)

llm = BedrockChat(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs={"temperature":0.0},
)

memory = ConversationBufferMemory(return_messages=True, memory_key="history")
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="あなたのタスクはユーザーの質問に明確に答えることです。"),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("""{input}""")
])

# モデル呼び出し
# 実行中しか過去の会話がメモリに残らないので、メモリに残すためにはメモリに保存する必要がある
llm_chain = ConversationChain(llm=llm, prompt=prompt, memory=memory)
result = llm_chain.run("なぜ空は青いのですか？")
print(result)

print("2つ目の質問")
result = llm_chain.run("1つ前にした質問文を出力してください。")
print(result)