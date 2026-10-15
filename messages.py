from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

messages = {
    'messages':
        [
            HumanMessage(
                content='Python 入门需要学习的知识点',
                additional_kwargs={},
                response_metadata={},
                id='391a46e7-210c-4b41-9f2c-c5b2896d902c'
            ),
            AIMessage(
                content='',
                additional_kwargs={'refusal': None},
                response_metadata={
                    'token_usage': {'completion_tokens': 15, 'prompt_tokens': 271, 'total_tokens': 286,
                                    'completion_tokens_details': None, 'prompt_tokens_details': None},
                    'model_provider': 'openai', 'model_name': 'glm-4-flash', 'system_fingerprint': None,
                    'id': '2026101417044941324b5314f244bb', 'finish_reason': 'tool_calls', 'logprobs': None
                },
                id='lc_run--01a139a8-31c2-7c83-9b2f-cb147a44d5f8-0',
                tool_calls=[
                    {'name': 'search_kb', 'args': {'query': 'Python 入门需要学习的知识点'},
                     'id': 'call_-7181089592680900018', 'type': 'tool_call'}
                ],
                invalid_tool_calls=[],
                usage_metadata={'input_tokens': 271, 'output_tokens': 15, 'total_tokens': 286,
                                'input_token_details': {}, 'output_token_details': {}}
            ),
            ToolMessage(
                content='[来源1] Python 入门课程\nPython 入门课程介绍变量、数据类型、条件判断、循环、函数和模块。',
                name='search_kb',
                id='97a18b90-ec16-4a04-b161-f09bac049ecc',
                tool_call_id='call_-7181089592680900018'
            ),
            AIMessage(
                content='Python 入门需要学习的知识点包括变量、数据类型、条件判断、循环、函数和模块等。这些内容可以在 RUNOOB 的 Python 入门课程中学习到。',
                additional_kwargs={'refusal': None},
                response_metadata={
                    'token_usage': {'completion_tokens': 39, 'prompt_tokens': 316, 'total_tokens': 355,
                                    'completion_tokens_details': None, 'prompt_tokens_details': None},
                    'model_provider': 'openai', 'model_name': 'glm-4-flash', 'system_fingerprint': None,
                    'id': '20261014170451a820873721b24067', 'finish_reason': 'stop', 'logprobs': None
                },
                id='lc_run--01a139a8-3a20-76e0-aae3-71b653483e95-0',
                tool_calls=[],
                invalid_tool_calls=[],
                usage_metadata={
                    'input_tokens': 316, 'output_tokens': 39, 'total_tokens': 355, 'input_token_details': {},
                    'output_token_details': {}
                }
            )
        ]
}
