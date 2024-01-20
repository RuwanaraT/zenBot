import datetime

def wake_up(text, name):
    return True if name in text.lower() else False

def action_time():
    return datetime.datetime.now().time().strftime('%H:%M')
