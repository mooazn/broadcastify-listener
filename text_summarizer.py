import queue
import requests
from bs4 import BeautifulSoup

q = queue.Queue()

def process_text_worker():
    while True:
        text = q.get()
        process_text(text)
        q.task_done()

def process_text(text):
    the_date = text[0]
    full_text = text[1]
    print(the_date, full_text)

def get_feed_location(feed_id):
    url = f"https://www.broadcastify.com/listen/feed/{feed_id}"
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")

    title = soup.find("title").text
    details = soup.find("span", class_="rrfont").text.strip()

    return title, details
