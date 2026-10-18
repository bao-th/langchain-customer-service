from dataclasses import dataclass

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import interrupt

from KnowledgeBase import KnowledgeBase
from knowledge_base import retriever

kb = KnowledgeBase('./my_knowledge_db')


@tool
def search_kb(query: str) -> str:
    """检索模拟课程知识库，查询教程、学习内容和系统说明。
    :param query: 用户查询问题
    :return: 知识库返回的检索结果
    """
    docs = kb.search(query, k=3)
    if not docs:
        return "知识库中没有检索到相关资料。"

    results = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        source = meta.get("source", "未知文档")
        page = meta.get("page")
        location = f"{source} / 第{page + 1}页" if page is not None else source
        results.append(f"[来源{i}] {location}\n{doc.page_content[:1500]}")

    return "\n\n".join(results)


# @tool
# def search_kb(query: str) -> str:
#     """检索模拟课程知识库，查询教程、学习内容和系统说明。
#     :param query: 用户查询问题
#     :return: 知识库返回的检索结果
#     """
#     docs = retriever.invoke(query)
#     if not docs:
#         return "未找到相关资料"
#     return '\n\n'.join(
#         f"[来源{i}] {doc.metadata.get('title', '未知')}\n"
#         f"{doc.page_content}"
#         for i, doc in enumerate(docs, 1)
#     )


orders_db = {
    "ORD-DEMO-001": {
        "owner_id": "user_xiaoming",
        "item": "Python 模拟练习课程",
        "amount": 99,
        "status": "已完成"
    },
    "ORD-DEMO-002": {
        "owner_id": "user_other",
        "item": "Java 模拟练习课程",
        "amount": 129,
        "status": "待处理"
    }
}


@dataclass
class UserContext:
    user_id: str


@tool
def query_order(order_id: str, runtime: ToolRuntime[UserContext]) -> str:
    """查询当前已认证用户的模拟订单。"""
    order_id = order_id.strip().upper()
    order = orders_db.get(order_id)

    if not order or order["owner_id"] != runtime.context.user_id:
        return "未找到可访问的订单。"

    return (
        f"模拟订单 {order_id}：{order['item']}，"
        f"¥{order['amount']}，状态：{order['status']}。"
    )


@tool
def transfer_to_human(reason: str) -> str:
    """请求模拟人工转接，需要外部审批。"""
    decision = interrupt({
        'action': 'transfer_to_human',
        'reason': reason,
        'message': '是否批准这次模拟人工转接？'
    })

    if not isinstance(decision, dict) or decision.get('approved') is not True:
        return '人工转接申请已取消。'

    return '模拟转接已获批准。本程序未连接真实人工客服系统，也没有创建工单。'


if __name__ == '__main__':
    print(search_kb.invoke('Python 入门需要学习的知识点'))
