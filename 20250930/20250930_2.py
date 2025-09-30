from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
import os
from langchain_core.vectorstores import InMemoryVectorStore
import asyncio

file_path = "../../../140120250724519337.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))

print(f"{docs[0].page_content[:200]}\n")
print(docs[0].metadata)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION_EMBEDDING"],
)

vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])

vector_store = InMemoryVectorStore(embeddings)

ids = vector_store.add_documents(documents=all_splits)

results = vector_store.similarity_search(
    "今期の営業利益は？"
)

print(results[0])

# 非同期関数として実行
#async def main():
#    results = await vector_store.asimilarity_search("セグメント別売上は？")
#    print(results[0])

# 非同期関数を実行
#asyncio.run(main())

# Note that providers implement different scores; the score here
# is a distance metric that varies inversely with similarity.

results = vector_store.similarity_search_with_score("セグメント別売上は？")
doc, score = results[0]
print(f"Score: {score}\n")
print(doc)

embedding = embeddings.embed_query("マイナンバー等のソーシャルDX")

results = vector_store.similarity_search_by_vector(embedding)
print(results[0])