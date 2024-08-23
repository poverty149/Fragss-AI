from flask import Flask,request,redirect,render_template, send_file
from werkzeug.utils import secure_filename
import os
from process_clips import *
from shot_detection import *
from video_enhancement import *

app=Flask(__name__)
app.config['UPLOAD_FOLDER']='./uploads/'
app.config['ALLOWED_EXTENSIONS']={'mp4','avi'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/',methods=['GET','POST'])
def upload_video():
    if request.method=='POST':
        print(request.form,"PP")
        if 'file' not in request.files:
            return redirect(request.url)
        if 'subtitles' not in request.form:
            sub=False
        else:
            sub=True
        file=request.files['file']

        
        if file and allowed_file(file.filename):
            filename=secure_filename(file.filename)
            filepath=os.path.join(app.config['UPLOAD_FOLDER'],filename)
            file.save(filepath)
        # print(filepath)
        frames, shot_boundaries = parallel_frame_processing(filepath)
        save_clips_using_processed_frames_with_audio(frames, shot_boundaries, filepath)
        output_dir='static/clips/subtitles'
        virality_scores=process_clips_from_folder('./static/clips')
        if sub:
            for clip,_ in virality_scores:
                audio=clip.replace('mp4','wav').replace('final_','')
                language,segments=transcribe('static/clips/'+audio)

                add_subtitle_to_video('./static/clips/'+clip,segments,output_dir)


        return render_template('result.html', virality_scores=virality_scores,sub=sub)

     

    return render_template('upload.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)


if __name__=="__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(port=8008,debug=True)