from langchain.agents.middleware import before_model, after_agent
from langchain_core.messages import AIMessage


@before_model(can_jump_to=['end'])
def content_guard(state, runtime):
    '''拦截测试命令，不执行模型。'''
    users = [msg for msg in state.get('messages', []) if msg.type == 'human']
    if users and users[-1].content.strip() == 'BLOCK_DEMO':
        return {
            'jump_to': 'end',
            'messages': [AIMessage(content='这是一条演示拦截结果。')]
        }
    return None


SIGNATURE = "\n\n---\nRUNOOB AI 客服演示｜非真实交易或人工客服服务"


@after_agent
def auto_signature(state, runtime):
    '''仅为最终ai文本回答追加一次签名SIGNATURE'''
    messages = state.get('messages', [])
    if not messages:
        return None

    last = messages[-1]
    if last.type != 'ai' or getattr(last, 'tool_calls', []):
        return None
    if not isinstance(last.content, str) or SIGNATURE in last.content:
        return None
    return {
        'messages': [last.model_copy(update={'content': last.content + SIGNATURE})]
    }
