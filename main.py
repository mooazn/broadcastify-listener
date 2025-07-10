from get_audio import AudioListener
from text_summarizer import process_text_worker, get_feed_location
import threading

stream_id = '43934'
threading.Thread(target=process_text_worker, daemon=True).start()
print(get_feed_location(stream_id))

AudioListener(f'https://broadcastify.cdnstream1.com/{stream_id}')
