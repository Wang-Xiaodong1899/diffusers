
from pytubefix import YouTube
from pytubefix.cli import on_progress
 
url = "https://www.youtube.com/watch?v=6YiU1PLeifE"

yt = YouTube(url, use_oauth=True, allow_oauth_cache=True, on_progress_callback = on_progress)
ys = yt.streams.get_highest_resolution()

ys.download() # you will only get the request to authenticate once you download
