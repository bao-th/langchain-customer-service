import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

from env_utils import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL


class KnowledgeBase:
    """支持文件增量更新的个人知识库。"""

    def __init__(self, persist_dir="./my_knowledge_db"):
        self.embeddings = OpenAIEmbeddings(
            model='text-embedding-v4',
            api_key=SecretStr(DASHSCOPE_API_KEY),
            base_url=DASHSCOPE_BASE_URL,
            check_embedding_ctx_length=False,
            chunk_size=10,
        )

        self.vector_store = Chroma(
            collection_name='personal_knowledge_v1',
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            add_start_index=True
        )

    def _source_key(self, path: Path) -> str:
        """根据文件路径生成唯一键。"""
        return hashlib.sha256(str(path.resolve()).encode('utf-8')).hexdigest()

    def _existing_ids(self, source_key: str) -> list:
        """获取已存在的向量ID。"""
        data = self.vector_store.get(where={'source_key': source_key})
        return data['ids']

    def add_file(self, file_path: str) -> int:
        """导入新文件；相同版本跳过，修改后更新。"""
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)

        if path.suffix.lower() not in [".txt", ".md", ".pdf"]:
            raise ValueError("仅支持 TXT、Markdown 和 PDF")

        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        source_key = self._source_key(path)
        old_ids = self._existing_ids(source_key)

        # 同一版本已建索引，直接跳过
        prefix = f"{source_key}-{file_hash}-"
        if old_ids and all(i.startswith(prefix) for i in old_ids):
            print(f"[跳过] {path.name} 未发生变化")
            return 0

        loader = PyPDFLoader(str(path)) if path.suffix.lower() == '.pdf' else TextLoader(str(path),
                                                                                         encoding='utf-8')
        docs = loader.load()

        for doc in docs:
            doc.metadata.update({
                'source': path.name,
                'source_path': str(path),
                'source_key': source_key,
                'file_hash': file_hash,
            })

        chunks = [d for d in self.splitter.split_documents(docs) if d.page_content.strip()]
        if not chunks:
            raise ValueError(f"没有提取到可索引的文字：{path.name}，扫描件可能需要 OCR")

        new_ids = [f"{source_key}-{file_hash}-{i}" for i in range(len(chunks))]
        for i, doc in enumerate(chunks):
            doc.metadata['chunk_index'] = i

        # 先写入新版本，成功后再删除旧版本
        self.vector_store.add_documents(chunks, ids=new_ids)
        obsolete = list(set(old_ids) - set(new_ids))
        if obsolete:
            self.vector_store.delete(ids=obsolete)

        print(f"[已更新] {path.name}: {len(chunks)} 个片段")
        return len(chunks)

    def add_text(self, text: str, source: str = 'manual') -> int:
        """按来源名称更新手动文本。"""
        if not text.strip():
            raise ValueError("不能添加空文本")

        source_key = "manual-" + hashlib.sha256(source.encode()).hexdigest()
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        old_ids = self._existing_ids(source_key)

        prefix = f"{source_key}-{content_hash}-"
        if old_ids and all(i.startswith(prefix) for i in old_ids):
            print(f"[跳过] {source} 未发生变化")
            return 0

        chunks = self.splitter.create_documents(
            [text],
            metadatas=[{'source': source, 'source_key': source_key, 'file_hash': content_hash}],
        )

        new_ids = [f"{source_key}-{content_hash}-{i}" for i in range(len(chunks))]
        for i, doc in enumerate(chunks):
            doc.metadata['chunk_index'] = i

        self.vector_store.add_documents(chunks, ids=new_ids)
        obsolete = list(set(old_ids) - set(new_ids))
        if obsolete:
            self.vector_store.delete(ids=obsolete)

        return len(chunks)

    def remove_file(self, file_path: str) -> int:
        """从向量库中移除指定来源文件。"""
        source_key = self._source_key(file_path)
        ids = self._existing_ids(source_key)
        if ids:
            self.vector_store.delete(ids=ids)
        return len(ids)

    def search(self, query: str, k: int = 3) -> list:
        """执行语义检索。"""
        return self.vector_store.similarity_search(query, k=k)

    def get_retriever(self, k: int = 3):
        """返回 LangChain Retriever。"""
        return self.vector_store.as_retriever(search_kwargs={'k': k})


if __name__ == "__main__":
    # 创建知识库
    kb = KnowledgeBase("./my_knowledge_db")

    # 添加演示资料，重复执行会自动跳过相同版本
    kb.add_text(
        "Python 学习路线可以从基础语法开始，然后学习数据结构、Web 开发或数据分析，并通过项目练习巩固。",
        source="Python 学习路线"
    )
    kb.add_text(
        "函数用于封装重复代码。Python 通过 def 定义函数，可以接收参数并返回结果。",
        source="Python 函数笔记"
    )
    kb.add_text(
        "Python 入门课程介绍变量、数据类型、条件判断、循环、函数和模块。",
        source="Python 入门课程"
    )
    kb.add_text(
        "HTML 入门课程介绍网页结构、标签、链接、图片、列表、表格与表单。",
        source="HTML 入门课程"
    )
    kb.add_text(
        "Java 入门课程包含基本语法、类、对象、继承、接口和异常处理。",
        source="Java 入门课程"
    ),

    # 自动导入 docs 目录中的文档
    docs_dir = Path("./docs")
    if docs_dir.exists():
        for path in docs_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf"}:
                try:
                    kb.add_file(str(path))
                except Exception as e:
                    print(f"[跳过失败文档] {path.name}: {e}")
