# 会話履歴をDynamoDBに保存し、過去の会話を参照できるスクリプト
from langchain_core.messages import SystemMessage

from boto3.session import Session
from langchain.chat_models import BedrockChat
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

import os
from dotenv import load_dotenv

load_dotenv()

session_id = "<ENTER_YOUR_SESSION_ID>"
region = os.environ["AWS_REGION"]
table_name = os.environ["DYNAMO_TABLE_NAME"]
model_id = os.environ["MODEL_ID"]

session = Session(profile_name=os.environ["AWS_PROFILE"])
bedrock_runtime = session.client("bedrock-runtime", region_name=region)

llm = BedrockChat(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs={"temperature":0.0},
)

dynamodb_chat_history = DynamoDBChatMessageHistory(
    table_name=table_name,
    session_id=session_id
)

memory = ConversationBufferMemory(
    memory_key="history",
    chat_memory=dynamodb_chat_history,
    return_messages=True
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="あなたのタスクはユーザーの質問に明確に答えることです。"),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("""{input}""")
])

llm_chain = ConversationChain(llm=llm, prompt=prompt, memory=memory)
result = llm_chain.invoke("日本で一番高い山は何ですか？")
# result = llm_chain.invoke("2番目は何ですか？")
print(result["response"])