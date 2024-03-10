from flask import Flask, flash, request, redirect, url_for, render_template,send_file,session
import urllib.request
import os
from werkzeug.utils import secure_filename
from hybridmodel import predict1 
from shutil import copyfile

 
app = Flask(__name__)
 
UPLOAD_FOLDER = os.path.join(r"D:\projects\optimized hybrid deep grid model\techgium_ui\techgium\static", 'images')
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
        
 
@app.route('/')
def home():
    return render_template('index.html') 
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        #print('upload_image filename: ' + filename)
        prediction=(predict1(filename))
        # print("app.w=",hybridmodel.w,"Pred",prediction)
        print("app",prediction)
        if(prediction==0):
            return render_template('ndn.html',file=filename)   
        else:
            return render_template('dn.html',file=filename)            

     
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='images/' + filename), code=301)

@app.route('/download_file')
def download_file():
    path = r"D:\projects\optimized hybrid deep grid model\techgium_ui\techgium\x-ray_without_gridlines.jpg"
    return send_file(path,as_attachment=True)

@app.route('/show_file')
def show_file():
    path =os.path.join(session['uploaded_img_file_path'] )
    return send_file(path,as_attachment=True)
 
 
if __name__ == "__main__":
    app.run()