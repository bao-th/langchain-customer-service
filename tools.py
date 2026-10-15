from dataclasses import dataclass

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

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
    order = orders_db[order_id]

    if not order or order["owner_id"] != runtime.context.user_id:
        return "未找到可访问的订单。"

    return (
        f"模拟订单 {order_id}：{order['item']}，"
        f"¥{order['amount']}，状态：{order['status']}。"
    )


if __name__ == '__main__':
    print(search_kb.invoke('Python 入门需要学习的知识点'))
