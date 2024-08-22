from flask import Flask,request,redirect,render_template, send_file
from werkzeug.utils import secure_filename
import os


app=Flask(__name__)
app.config['UPLOAD_FOLDER']='./uploads/'
app.config['ALLOWED_EXTENSIONS']={'mp4','avi'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
def process_clips(filepath):
    clip_scores="clips/output.mp4"
    virality={'final_clip_13.mp4': 8, 'final_clip_15.mp4': 6}
    return  clip_scores,virality
@app.route('/',methods=['GET','POST'])
def upload_video():
    if request.method=='POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file=request.files['file']

        
        if file and allowed_file(file.filename):
            filename=secure_filename(file.filename)
            filepath=os.path.join(app.config['UPLOAD_FOLDER'],filename)
            file.save(filepath)
        print(filepath)
        clip_scores,virality_scores=process_clips(filepath)

        return render_template('result.html',clip_output=clip_scores, virality_scores=virality_scores)

     

    return render_template('upload.html')
@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)



if __name__=="__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(port=5000,debug=True)