# 実行中しか会話履歴が残らないスクリプト

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_aws import ChatBedrock
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from boto3.session import Session
import os
from dotenv import load_dotenv

load_dotenv()

region = os.environ["AWS_REGION"]
model_id = os.environ["MODEL_ID"]

session = Session(
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=region
)

bedrock_runtime = session.client("bedrock-runtime", region_name=region)

llm = ChatBedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs={"temperature": 0.0},
    region_name=region
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="あなたのタスクはユーザーの質問に明確に答えることです。"),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# 会話履歴をリストで管理
history = []

def chat_with_memory(input_text):
    global history
    formatted_prompt = prompt.invoke({"history": history, "input": input_text})
    
    response = llm.invoke(formatted_prompt)
    
    # 会話履歴を更新
    history.append(HumanMessage(content=input_text))
    history.append(AIMessage(content=response.content))
    
    return response.content

# モデル呼び出し
result = chat_with_memory("なぜ空は青いのですか？")
print(result)

print("2つ目の質問")
result = chat_with_memory("1つ前にした質問文を出力してください。")
print(result)
