import requests
from bs4 import BeautifulSoup
import re
import csv
import os 
import yt_dlp
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

# Function to download videos
def download_videos(url_name_list, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for url, custom_name in url_name_list:
        ydl_opts = {
            'outtmpl': os.path.join(output_directory, f'{custom_name}.%(ext)s'),
            'progress_hooks': [dl_progress_hook],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info_dict = ydl.extract_info(url, download=True)
                print(info_dict)
                return info_dict
                break
                video_title = info_dict.get('title', None)
                print(f'The video "{video_title}" has been downloaded and saved as "{custom_name}" on the desktop.')
            except Exception as e:
                print(f"Error downloading video {url}: {e}")
def get_video_metadata(video_url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    response = requests.get(video_url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to retrieve the video page. Status code: {response.status_code}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    metadata = {}

    # Extract video ID from the URL
    video_id_match = re.search(r"v=([a-zA-Z0-9_-]+)", video_url)
    metadata['id'] = video_id_match.group(1) if video_id_match else 'N/A'

    # Get views
    views = soup.find('meta', itemprop='interactionCount')
    metadata['views'] = views['content'] if views else 'N/A'

    # Get likes using regex
    likes_match = re.search(r'"label":"(\d+\.?\d*[KM]?) likes"', response.text)
    if likes_match:
        metadata['likes'] = likes_match.group(1)
    else:
        # Attempt to find likes in the JSON-like structure
        likes_accessibility_match = re.search(r'"accessibilityText":"like this video along with (\d+(,\d+)*) other people"', response.text)
        if likes_accessibility_match:
            likes = likes_accessibility_match.group(1).replace(',', '')
            metadata['likes'] = likes
        else:
            metadata['likes'] = 'N/A'

    # Get shares (not directly available, set as N/A)
    metadata['shares'] = 'N/A'
        # Get duration
    duration_match = re.search(r'"lengthSeconds":"(\d+)"', response.text)
    metadata['duration'] = duration_match.group(1) if duration_match else 'N/A'

    # Get description
    description = soup.find('meta', property='og:description')
    metadata['description'] = description['content'] if description else 'N/A'

    # Get video URL
    metadata['url'] = video_url

    return metadata

def write_metadata_to_csv(metadata_list, filename):
    fields = ['id', 'views', 'likes', 'shares', 'duration', 'description', 'url']
    with open(filename, 'w', newline='') as f:
        csvfile = csv.DictWriter(f, delimiter=',', fieldnames=fields)
        csvfile.writeheader()
        csvfile.writerows(metadata_list)

if __name__=='__main__':
    output_directory="clips/trial"
    video_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Replace with actual YouTube video URLs
            # Add more URLs as needed
        ]

    metadata_list = []
    url_name_list=[]
    for url in video_urls:
        metadata = get_video_metadata(url)
        if metadata:
            metadata_list.append(metadata)
        url_name_list.append((url,metadata['id']))

    if metadata_list:
        write_metadata_to_csv(metadata_list, 'youtube_videos_metadata.csv')
        print("Metadata written to youtube_videos_metadata.csv")
    else:
        print("No metadata to write.")
    download_videos(url_name_list,output_directory)