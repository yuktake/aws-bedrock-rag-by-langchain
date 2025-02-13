# 会話履歴をDynamoDBに保存し、過去の会話を参照できるスクリプト
from boto3.session import Session
from langchain_aws import ChatBedrock
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

import os
from dotenv import load_dotenv

load_dotenv()

session_id = "<ENTER_YOUR_SESSION_ID>"
region = os.environ["AWS_REGION"]
table_name = os.environ["DYNAMO_TABLE_NAME"]
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
    model_kwargs={"temperature":0.0},
    region_name = region
)

# langchain_community.chat_message_historiesからはbedrockとdynamodbのリージョンを揃える必要があった。
dynamodb_chat_history = DynamoDBChatMessageHistory(
    table_name=table_name,
    session_id=session_id,
    boto3_session=session,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたのタスクはユーザーの質問に明確に答えることです。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm
llm_chain = RunnableWithMessageHistory(
    chain,
    get_session_history=lambda session_id: dynamodb_chat_history,
    input_messages_key="input",
    history_messages_key="history"
)

result = llm_chain.invoke(
    {"input": "日本で一番高い山は何ですか？"}, 
    # {"input": "2番目は何ですか？"}, 
    config={
        "configurable": {"session_id": session_id}
    }
)
print(result)