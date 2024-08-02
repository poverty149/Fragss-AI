import os
import time
import math
import ffmpeg
import logging
from faster_whisper import WhisperModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

input_video = "new_seg.mp4"
input_video_name = os.path.splitext(input_video)[0]

def extract_audio(input_video):
    try:
        extracted_audio = f"audio-{input_video_name}.wav"
        stream = ffmpeg.input(input_video)
        stream = ffmpeg.output(stream, extracted_audio)
        ffmpeg.run(stream, overwrite_output=True)
        logging.info(f"Audio extracted: {extracted_audio}")
        return extracted_audio
    except ffmpeg.Error as e:
        logging.error(f"Error extracting audio: {e}")
        return None

def transcribe(audio):
    try:
        model = WhisperModel("small")
        segments, info = model.transcribe(audio)
        language = info[0]
        logging.info(f"Transcription language: {language}")
        segments = list(segments)
        for segment in segments:
            logging.info(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        return language, segments
    except Exception as e:
        logging.error(f"Error in transcription: {e}")
        return None, None

def format_time(seconds):
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    return formatted_time

def generate_subtitle_file(language, segments):
    try:
        subtitle_file = f"sub-{input_video_name}.{language}.srt"
        text = ""
        for index, segment in enumerate(segments):
            segment_start = format_time(segment.start)
            segment_end = format_time(segment.end)
            text += f"{index + 1}\n"
            text += f"{segment_start} --> {segment_end}\n"
            text += f"{segment.text}\n\n"
        with open(subtitle_file, "w") as f:
            f.write(text)
        logging.info(f"Subtitle file generated: {subtitle_file}")
        return subtitle_file
    except Exception as e:
        logging.error(f"Error generating subtitle file: {e}")
        return None

def add_subtitle_to_video(soft_subtitle, subtitle_file, subtitle_language):
    try:
        video_input_stream = ffmpeg.input(input_video)
        subtitle_input_stream = ffmpeg.input(subtitle_file)
        output_video = f"output-{input_video_name}.mp4"
        subtitle_track_title = os.path.splitext(subtitle_file)[0]

        if soft_subtitle:
            stream = ffmpeg.output(
                video_input_stream, subtitle_input_stream, output_video,
                **{"c": "copy", "c:s": "mov_text"},
                **{"metadata:s:s:0": f"language={subtitle_language}", "metadata:s:s:0": f"title={subtitle_track_title}"}
            )
        else:
            stream = ffmpeg.output(video_input_stream, output_video, vf=f"subtitles={subtitle_file}")
        
        ffmpeg.run(stream, overwrite_output=True)
        logging.info(f"Output video created: {output_video}")
    except ffmpeg.Error as e:
        logging.error(f"Error adding subtitle to video: {e}")

def enhance_video_with_aspect_ratio(input_video, output_video, width=None, height=None):
    try:
        if width:
            stream = ffmpeg.input(input_video)
            stream = ffmpeg.output(stream, output_video, vf=f"scale={width}:-1")
        elif height:
            stream = ffmpeg.input(input_video)
            stream = ffmpeg.output(stream, output_video, vf=f"scale=-1:{height}")
        else:
            logging.error("Either width or height must be specified for aspect ratio adjustment.")
            return None

        ffmpeg.run(stream, overwrite_output=True)
        logging.info(f"Video enhanced with aspect ratio adjustment: {output_video}")
        return output_video
    except ffmpeg.Error as e:
        logging.error(f"Error enhancing video: {e}")
        return None

if __name__ == "__main__":
    extracted_audio = extract_audio(input_video)
    if extracted_audio:
        language, segments = transcribe(extracted_audio)
        if language and segments:
            subtitle_file = generate_subtitle_file(language, segments)
            if subtitle_file:
                enhanced_video = enhance_video_with_aspect_ratio(input_video, f"enhanced-{input_video_name}.mp4", width=1280)
                if enhanced_video:
                    add_subtitle_to_video(soft_subtitle=False, subtitle_file=subtitle_file, subtitle_language=language)
