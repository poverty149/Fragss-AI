import os
import math
import ffmpeg
import logging
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ColorClip
from faster_whisper import WhisperModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

input_video = "sample2.mp4"
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
        segments, info = model.transcribe(audio, word_timestamps=True)
        language = info[0]
        logging.info(f"Transcription language: {language}")
        segments = list(segments)
        for segment in segments:
            logging.info(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        return language, segments
    except Exception as e:
        logging.error(f"Error in transcription: {e}")
        return None, None

def create_caption(segment, framesize, font="A", color='white', highlight_color='yellow',highlight_bg='red', stroke_color='black', stroke_width=1.5):
    word_clips = []
    xy_textclips_positions = []

    x_pos = 0
    y_pos = 0
    line_width = 0  # Total width of words in the current line
    frame_width = framesize[0]
    frame_height = framesize[1]

    x_buffer = frame_width * 1 / 10
    max_line_width = frame_width - 2 * x_buffer
    fontsize = int(frame_height * 0.075)  # 7.5 percent of video height

    for word in segment.words:
        duration = word.end - word.start
        word_clip = TextClip(word.word, font=font, fontsize=fontsize, color=color, stroke_color=stroke_color, stroke_width=stroke_width).set_start(segment.start).set_duration(segment.end - segment.start)
        word_width, word_height = word_clip.size
        
        if line_width + word_width <= max_line_width:
            xy_textclips_positions.append({
                "x_pos": x_pos,
                "y_pos": y_pos,
                "width": word_width,
                "height": word_height,
                "word": word.word,
                "start": word.start,
                "end": word.end,
                "duration": duration
            })

            word_clip = word_clip.set_position((x_pos, y_pos))
            x_pos = x_pos + word_width
            line_width = line_width + word_width
        else:
            x_pos = 0
            y_pos = y_pos + word_height + 10
            line_width = word_width

            xy_textclips_positions.append({
                "x_pos": x_pos,
                "y_pos": y_pos,
                "width": word_width,
                "height": word_height,
                "word": word.word,
                "start": word.start,
                "end": word.end,
                "duration": duration
            })

            word_clip = word_clip.set_position((x_pos, y_pos))
            x_pos = word_width

        word_clips.append(word_clip)

    for highlight_word in xy_textclips_positions:
        word_clip_highlight = TextClip(highlight_word['word'], font=font, fontsize=fontsize, color=highlight_color,bg_color=highlight_bg, stroke_color=stroke_color, stroke_width=stroke_width).set_start(highlight_word['start']).set_duration(highlight_word['duration'])
        word_clip_highlight = word_clip_highlight.set_position((highlight_word['x_pos'], highlight_word['y_pos']))
        word_clips.append(word_clip_highlight)

    return word_clips, xy_textclips_positions

def add_subtitle_to_video(input_video, segments):
    try:
        video = VideoFileClip(input_video)
        frame_size = video.size

        all_linelevel_splits = []

        for segment in segments:
            out_clips, positions = create_caption(segment, frame_size)

            max_width = 0
            max_height = 0

            for position in positions:
                x_pos, y_pos = position['x_pos'], position['y_pos']
                width, height = position['width'], position['height']

                max_width = max(max_width, x_pos + width)
                max_height = max(max_height, y_pos + height)

            color_clip = ColorClip(size=(int(max_width * 1.1), int(max_height * 1.1)), color=(64, 64, 64))
            color_clip = color_clip.set_opacity(.6)
            color_clip = color_clip.set_start(segment.start).set_duration(segment.end - segment.start)

            clip_to_overlay = CompositeVideoClip([color_clip] + out_clips)
            clip_to_overlay = clip_to_overlay.set_position("bottom")

            all_linelevel_splits.append(clip_to_overlay)

        final_video = CompositeVideoClip([video] + all_linelevel_splits)

        final_video = final_video.set_audio(video.audio)

        output_video = f"output-{input_video_name}.mp4"
        final_video.write_videofile(output_video, fps=24, codec="libx264", audio_codec="aac")
        logging.info(f"Output video created: {output_video}")
    except Exception as e:
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

extracted_audio = extract_audio(input_video)
if extracted_audio:
    language, segments = transcribe(extracted_audio)
    if language and segments:
        enhanced_video = enhance_video_with_aspect_ratio(input_video, f"enhanced-{input_video_name}.mp4", width=1280)
        if enhanced_video:
            add_subtitle_to_video(enhanced_video, segments)
