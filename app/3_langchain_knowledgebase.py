# DynamooDBの過去の会話を参照しつつ、AWS Bedrockを利用して質問に答えるスクリプト
from boto3.session import Session
from langchain.memory import ConversationSummaryMemory
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from langchain_aws import AmazonKnowledgeBasesRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain_aws import ChatBedrock

import os
from dotenv import load_dotenv

load_dotenv()

session_id = "<ENTER_YOUR_SESSION_ID>"
region = os.environ["AWS_REGION"]
knowledge_base_id = os.environ["KNOWLEDGE_BASE_ID"]
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

retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id=knowledge_base_id,
    retrieval_config={
        "vectorSearchConfiguration": {
            "numberOfResults": 4
        }
    },
    # NOTE:セッションでリージョンを指定するだけではエラーが発生するため、リージョンを明示的に指定する
    region_name=region
)

dynamodb_chat_history = DynamoDBChatMessageHistory(
    table_name=table_name,
    session_id=session_id,
    boto3_session=session
)

# ConversationSummaryMemoryを使うことで、過去の会話履歴をllmによって要約して渡すことができる
# ConversationBufferWindowMemoryの場合は、過去の会話履歴をそのまま渡すことができるが、渡すデータ量を制限することができる
memory = ConversationSummaryMemory(
    llm=llm,  # 要約を生成するための LLM が必要
    memory_key="chat_history",
    chat_memory=dynamodb_chat_history,
    return_messages=True
)

# ConversationalRetrievalChainの構築
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    # ドキュメントにはmemory引数がないが、なぜ指定できる？
    memory=memory,
)

# 「最初の質問文を出力してください」と質問した時に、正しく過去の質問の答え（日本で一番高い塔の話）
# が返ってくることを確認できたので、過去の会話履歴を取得しながらS3を参照できていることがわかる
result = qa_chain.invoke("福岡市の酒造を教えてください。")

print(result["answer"])