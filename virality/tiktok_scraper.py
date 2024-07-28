from TikTokApi import TikTokApi
import asyncio
import os
import requests
import csv
import csv
import yt_dlp
import pandas as pd

ms_token = 'write your ms_token here. Use inspect and go to cookies'

def dl_progress_hook(d):
    if d['status'] == 'downloading':
        downloaded = d.get('downloaded_bytes', 0)
        total = d.get('total_bytes', 1)
        percentage = downloaded / total * 100

        speed = d.get('speed', 0)
        speed_str = f'{speed:.2f}' if speed is not None else 'Unknown'

        eta = d.get('eta', 0)
        eta_str = f'{eta}s' if eta is not None else 'Unknown'

        print(f'Downloading: {percentage:.2f}% at {speed_str} bytes/s, ETA: {eta_str}')

def download_videos(url_name_list, output_directory):
    for url, custom_name in url_name_list:
        ydl_opts = {
            'outtmpl': os.path.join(output_directory, f'{custom_name}.%(ext)s'),
            'progress_hooks': [dl_progress_hook],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info_dict = ydl.extract_info(url, download=True)
                video_title = info_dict.get('title', None)
                print(f'The video "{video_title}" has been downloaded and saved as "{custom_name}" on the desktop.')
            except Exception as e:
                print(f"Error downloading video {url}: {e}")


async def get_related_videos_metadata():
    async with TikTokApi() as api:
        await api.create_sessions(headless=False, ms_tokens=[ms_token], num_sessions=1, sleep_after=3)
        
        # Starting video URL (Valorant video example)
        video_url = "https://www.tiktok.com/@akmancodm/video/7391508217165122849?lang=en"
        video = api.video(url=video_url)
        
        # Fetch related videos
        related_videos_metadata = []
        url_name_list = []

        async for related_video in video.related_videos(count=20):
            related_video_dict = related_video.as_dict
            related_video_url = f"https://www.tiktok.com/@{related_video_dict['author']['uniqueId']}/video/{related_video_dict['id']}"
            related_video = api.video(url=related_video_url)  # Set the URL for each related video
            video_info = await related_video.info()  # Fetch video information
            video_metadata = {
                'id': video_info['id'],
                'views': video_info['stats']['playCount'],
                'likes': video_info['stats']['diggCount'],
                'shares': video_info['stats']['commentCount'],
                'duration': video_info['video']['duration'],
                'description': video_info['desc'],
                'url': related_video_url
            }
            related_videos_metadata.append(video_metadata)
            url_name_list.append((related_video_url, f"{video_info['id']}"))

        # Write to CSV
        fields = ['id', 'views', 'likes', 'shares', 'duration', 'description', 'url']
        with open("cod_videos_meta.csv", 'w', newline='') as f:
            csvfile = csv.DictWriter(f, delimiter=',', fieldnames=fields)
            csvfile.writeheader()
            csvfile.writerows(related_videos_metadata)

        # Download videos
        output_directory = "./clips/Viral"
        download_videos(url_name_list, output_directory)

        return related_videos_metadata


if __name__ == "__main__":
    # Run the async function
    asyncio.run(get_related_videos_metadata())