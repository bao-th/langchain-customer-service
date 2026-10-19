from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from env_utils import ZHIPU_API_KEY, ZHIPU_BASE_URL
from middlewares import content_guard, auto_signature, inject_date_context, SYSTEM_PROMPT
from tools import search_kb, query_order, UserContext, transfer_to_human, get_weather, query_schedule, send_email

# model = ChatDeepSeek(model='deepseek-chat', temperature=0)
# model = ChatOpenAI(
#     model='glm-4-flash',
#     temperature=0,
#     api_key=ZHIPU_API_KEY,
#     base_url=ZHIPU_BASE_URL
# )
model = init_chat_model(
    model="glm-4-flash",
    model_provider="openai",  # openai相当于使用ChatOpenAI，所以需要base_url
    temperature=0,
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL,
    max_tokens=128  # 新增这一行，限制最大输出token
)


def make_agent(checkpointer):
    return create_agent(
        model=model,
        tools=[search_kb, query_order, transfer_to_human, get_weather, query_schedule, send_email],
        middleware=[content_guard, auto_signature, inject_date_context],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,  # 保存对话记录
        context_schema=UserContext,
    )


def chat(agent, thread_id: str, user_id: str, user_input: str, checkpointer):
    config = {'configurable': {'thread_id': thread_id}}

    # 输入reset 就清空当前thread历史
    if user_input.strip().lower() == "reset":
        checkpointer.delete_thread(thread_id)
        return {
            "messages": [
                AIMessage(content="✅ 对话历史已清空，可以重新开始对话")
            ]
        }

    result = agent.invoke(
        {"messages": [{'role': 'user', 'content': user_input}]},
        config=config,
        context=UserContext(user_id=user_id)
    )

    pending = result.get('__interrupt__', [])
    # 存在中断：返回待审批状态，前端/控制台展示审批面板
    if pending:
        return {
            'status': 'pending_approval',
            'actions': [item.value for item in pending]
        }

    return result


def chat_stream(agent, thread_id: str, user_id: str, user_input: str, checkpointer):
    config = {'configurable': {'thread_id': thread_id}}

    result = agent.stream(
        {"messages": [{'role': 'user', 'content': user_input}]},
        config=config,
        context=UserContext(user_id=user_id),
        stream_mode="messages"
    )

    return result


# 恢复转接申请
def resume_transfer(agent, thread_id: str, user_id: str, approved: bool):
    config = {'configurable': {'thread_id': thread_id}}
    state = agent.get_state(config)
    pending = [i for task in state.tasks for i in task.interrupts]

    if not pending:
        return {'status': 'error', "message": "当前没有等待处理的转接申请。"}

    # 演示函数假定后台已验证审批权限和会话归属
    result = agent.invoke(
        Command(resume={'approved': approved}),
        config=config,
        context=UserContext(user_id=user_id)
    )

    if result.get('__interrupt__', ()):
        return {
            'status': 'pending_approval',
            'message': '仍有待处理的中断。'
        }

    return result


def ask():
    """对话"""
    with SqliteSaver.from_conn_string('./customer_service_db/customer_service.db') as checkpointer:
        agent = make_agent(checkpointer)
        thread_id = 'demo-customer-001'
        user_id = "user_xiaoming"
        while True:
            user_input = input("你：")
            if user_input.strip().lower() == "exit":
                print("会话结束")
                break

            res = chat(agent, thread_id, user_id, user_input, checkpointer)

            # 处理待审批状态
            if res.get('status') == 'pending_approval':
                print("待审批：", res["actions"])
                # 模拟管理员审批，
                choice = input("模拟管理员审批，输入 y 通过 / n 拒绝：").strip().lower()
                approved = choice == "y"
                res = resume_transfer(agent, thread_id, user_id, approved)

            # 打印AI最后一条回复
            print(f"AI：{res["messages"][-1].content}")

            # 遍历所有消息，打印类型+内容
            print("\033[33m===== 本轮完整消息列表 =====")
            for msg in res["messages"]:
                print(f"【{msg.type}】 {msg.content}")
            print("===== 结束结束结束结束 =====\n\033[0m")


def ask_stream():
    """流式对话"""
    with SqliteSaver.from_conn_string('./customer_service_db/customer_service.db') as checkpointer:
        agent = make_agent(checkpointer)
        thread_id = 'demo-customer-001'
        user_id = "user_xiaoming"
        while True:
            user_input = input("你：")
            if user_input.strip().lower() == "exit":
                print("会话结束", end='', flush=True)
                break

            res = chat_stream(agent, thread_id, user_id, user_input, checkpointer)
            print('AI：', end='', flush=True)

            for chunk, metadata in res:
                if metadata.get("langgraph_node") != "model":
                    continue
                if getattr(chunk, "tool_call_chunks", None):
                    continue
                if isinstance(chunk.content, str):
                    print(chunk.content, end='', flush=True)

            print('\n', flush=True)


if __name__ == '__main__':
    print("AI客服已启动，输入 exit 结束对话\n")
    ask()
    # ask_stream()
