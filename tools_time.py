from datetime import datetime
from zoneinfo import ZoneInfo

# Vivianna is local to a German-speaking household — temporal awareness is anchored
# to Europe/Berlin. The model renders the actual answer in the user's language; this
# only hands it an unambiguous real timestamp to reason from.
_TZ = ZoneInfo("Europe/Berlin")


def current_time_context() -> str:
    """One neutral reference line giving Vivianna real temporal awareness. Injected
    into the system prompt every turn so she can answer time/date questions correctly
    and reason about time-of-day (greetings, lateness, bedtime) without a tool trigger.
    Framed so she does not recite it unprompted."""
    now = datetime.now(_TZ)
    stamp = now.strftime("%A %Y-%m-%d %H:%M")
    return (
        f"[Current local time (Europe/Berlin): {stamp}. This is the real current "
        f"date and time. Use it when the conversation calls for it — greetings, "
        f"whether it is late, scheduling, how long ago something was — but do not "
        f"announce it unprompted.]"
    )
