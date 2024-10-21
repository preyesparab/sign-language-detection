# utils.py

import datetime

def within_operating_hours():
    """Check if the current time is between 6 PM and 10 PM."""
    current_time = datetime.datetime.now().time()
    start_time = datetime.time(18, 0)  # 6 PM
    end_time = datetime.time(22, 0)    # 10 PM
    return start_time <= current_time <= end_time
