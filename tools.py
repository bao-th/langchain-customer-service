from langchain_core.tools import tool

from knowledge_base import retriever


@tool
def search_kb(query: str) -> str:
    """检索模拟课程知识库，查询教程、学习内容和系统说明。
    :param query: 用户查询问题
    :return: 知识库返回的检索结果
    """
    docs = retriever.invoke(query)
    if not docs:
        return "未找到相关资料"
    return '\n\n'.join(
        f"[来源{i}] {doc.metadata.get('title', '未知')}\n"
        f"{doc.page_content}"
        for i, doc in enumerate(docs, 1)
    )


if __name__ == '__main__':
    print(search_kb.invoke('Python 入门需要学习的知识点'))
