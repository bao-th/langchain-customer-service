from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TIMEZONE = ZoneInfo("Asia/Shanghai")


def today_str() -> str:
    return datetime.now(TIMEZONE).date().isoformat()


today = today_str()
tomorrow = (datetime.now(TIMEZONE).date() + timedelta(days=1)).isoformat()

weather_db = {
    "杭州": {"condition": "晴", "temp": 25, "humidity": 60},
    "北京": {"condition": "多云", "temp": 18, "humidity": 45},
    "上海": {"condition": "小雨", "temp": 22, "humidity": 80}
}

calendar_events = [
    {"id": 1, "date": today, "time": "10:00", "title": "团队周会", "duration": "1小时"},
    {"id": 2, "date": today, "time": "14:00", "title": "Python 学习", "duration": "2小时"},
    {"id": 3, "date": tomorrow, "time": "15:00", "title": "代码审查", "duration": "1.5小时"}
]
