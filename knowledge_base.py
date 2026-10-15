from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

from env_utils import DASHSCOPE_API_KEY

# 实列文本向量大模型
embeddings = OpenAIEmbeddings(
    model='text-embedding-v4',
    api_key=SecretStr(DASHSCOPE_API_KEY),
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    check_embedding_ctx_length=False,  # 跳过长度校验，直接送入模型；风险：文本超长时，模型会自动截断，丢失信息，或者直接报错。
    chunk_size=10
)

# 实例数据库
vector_store = Chroma(
    collection_name='runoob_customer_demo',
    embedding_function=embeddings,
    persist_directory='./runoob_kb'
)

# 创建检索器
retriever = vector_store.as_retriever(search_kwargs={'k': 1})

print("数据库检索器初始化完成\n")


def init_knowledge_base():
    # 模拟知识库
    knowledge_docs = [
        Document(
            page_content='Python 入门课程介绍变量、数据类型、条件判断、循环、函数和模块。',
            metadata={'source': 'demo-python', 'title': 'Python 入门课程'}
        ),
        Document(
            page_content='HTML 入门课程介绍网页结构、标签、链接、图片、列表、表格与表单。',
            metadata={'source': 'demo-html', 'title': 'HTML 入门课程'}
        ),
        Document(
            page_content='Java 入门课程包含基本语法、类、对象、继承、接口和异常处理。',
            metadata={'source': 'demo-java', 'title': 'Java 入门课程'}
        ),
        Document(
            page_content='本演示系统不提供真实的会员价格、退款规则和客服营业时间。如需准确政策，请查询官方业务系统。',
            metadata={'source': 'demo-policy', 'title': '演示系统说明'}
        ),
    ]
    # 实例切分器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )
    # 切分文档
    chunks = splitter.split_documents(knowledge_docs)

    # 文档向量存入数据库
    ids = [doc.metadata['source'] for doc in chunks]
    vector_store.add_documents(documents=chunks, ids=ids)


if __name__ == '__main__':
    init_knowledge_base()
    collection = vector_store._collection
    result = collection.get(include=["documents", "metadatas", "embeddings"])

    print("知识库内容：")
    for i in range(len(result["ids"])):
        print(f"\nID: {result['ids'][i]}")
        print(f"内容: {result['documents'][i]}")
        print(f"元数据: {result['metadatas'][i]}")
        print(f"向量embedding（前10个值）: {result['embeddings'][i][:10]}")
        print(f"向量维度: {len(result['embeddings'][i])}")
