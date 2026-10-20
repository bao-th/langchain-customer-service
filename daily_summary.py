from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field

from customer_service_bot import model
from simulated_data import today_str
from tools import get_weather, query_schedule


class DailySummary(BaseModel):
    """每日摘要结构"""
    date: str = Field(description="日期，格式 YYYY-MM-DD")
    city: str = Field(description="天气所在城市")
    weather_summary: str = Field(description="天气概述")
    event_count: int = Field(ge=0, description="日程数量")
    key_events: list[str] = Field(description="日程事项")
    suggestions: str = Field(description="基于已知数据给出的建议")


def to_markdown(summary: DailySummary) -> str:
    """将 DailySummary 转换为 Markdown。"""
    lines = [
        f"## {summary.date} 每日摘要",
        "",
        f"**城市：** {summary.city}",
        f"**天气：** {summary.weather_summary}",
        f"**日程数量：** {summary.event_count}",
        "",
        "### 日程安排",
    ]
    lines.extend(f"- {event}" for event in summary.key_events)
    if not summary.key_events:
        lines.append('- 暂无日程')

    lines.extend(["", f"**建议：** {summary.suggestions}"])
    return "\n".join(lines)


summary_agent = create_agent(
    model=model,
    tools=[],
    response_format=ToolStrategy(DailySummary),
    system_prompt=(
        "你是每日摘要生成器。只能根据提供的数据生成 DailySummary。"
        "必须准确计算日程数量，不得补充不存在的天气或日程。"
        "天气为模拟数据，必须明确标记。"
    )
)


def generate_daily_summary(city: str = '杭州'):
    """获取模拟数据并生成结构化每日摘要。"""

    date_str = today_str()

    weather = get_weather.invoke({"city": city})
    schedule = query_schedule.invoke({"date_str": date_str})

    result = summary_agent.invoke({
        'messages': [{
            'role': 'user',
            'content': (
                f"日期：{date_str}\n城市：{city}\n"
                f"天气资料：\n{weather}\n"
                f"日程资料：\n{schedule}\n"
                "请只根据上述资料生成结构化摘要和建议"
            )
        }]
    })

    summary = result['structured_response']
    return to_markdown(summary)


if __name__ == "__main__":
    print("\n=== 测试：结构化每日摘要 ===")
    print(generate_daily_summary("杭州"))
