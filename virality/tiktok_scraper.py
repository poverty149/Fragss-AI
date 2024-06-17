from TikTokApi import TikTokApi
import asyncio
import os
import requests
import csv

ms_token = 'write your ms_token here. Use inspect and go to cookies'

### Need to further edit the download method. Currently all downloads are failing and had to manually download the videos.
async def download_tiktok_video(video_info, save_path):
    download_url=video_info['video']['downloadAddr']
    response = requests.get(download_url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Video downloaded successfully and saved as {save_path}")
    else:
        print("Failed to download the video")

async def get_related_videos_metadata():
    async with TikTokApi() as api:
        await api.create_sessions(headless=False, ms_tokens=[ms_token], num_sessions=1, sleep_after=3)
        
        # Starting video URL (Valorant video example)
        video_url = "https://www.tiktok.com/@valorant.videos/video/7149159347409538305"
        video = api.video(url=video_url)
        
        # Fetch related videos
        related_videos_metadata = []

        async for related_video in video.related_videos(count=10):
            related_video_dict = related_video.as_dict
            related_video_url = f"https://www.tiktok.com/@{related_video_dict['author']['uniqueId']}/video/{related_video_dict['id']}"
            related_video = api.video(url=related_video_url)  # Set the URL for each related video
            video_info = await related_video.info()  # Fetch video information
            # comments = await related_video.comments(count=10)  # Fetch comments
            video_metadata = {
                'id': video_info['id'],
                'views': video_info['stats']['playCount'],
                'likes': video_info['stats']['diggCount'],
                # 'comments': [comment['text'] for comment in comments],
                'shares': video_info['stats']['commentCount'],
                'duration': video_info['video']['duration'],
                'description': video_info['desc'],
                'url': related_video_url
            }
            related_videos_metadata.append(video_metadata)
            save_path = f"related_video_{video_info['id']}.mp4"
            # await download_tiktok_video(video_info, save_path)

        # Write to CSV
        fields = ['id', 'views', 'likes', 'shares', 'duration', 'description', 'url']
        with open("related_videos_metadata.csv", 'w', newline='') as f:
            csvfile = csv.DictWriter(f, delimiter=',', fieldnames=fields)
            csvfile.writeheader()
            csvfile.writerows(related_videos_metadata)

        return related_videos_metadata

if __name__ == "__main__":
    # Run the async function
    asyncio.run(get_related_videos_metadata())
