import os
from flask import Flask, request, url_for, send_from_directory
from werkzeug import secure_filename
from model.Model import Model

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '{}/upload'.format(os.getcwd())
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

html = '''
    <!DOCTYPE html>
    <title>Dog Breed Detector</title>
    <h1>Dog Breed Detector</h1>
    <form method=post enctype=multipart/form-data>
         <input type=file name=file>
         <input type=submit value=upload>
    </form>
    '''

def allowed_file(filename):
    '''
        a function to check file type
    '''

    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
    
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    '''
        a function to show image
    '''

    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    '''
        a function to upload image file
    '''

    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_url = url_for('uploaded_file', filename=filename)
            modelObj = Model()
            modelObj.load()
            pred = modelObj.get_result(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return html + '<label>' +  pred + '</label><br><img src=' + file_url + '>'
    return html

if __name__ == '__main__':
    '''
        main run part
    '''

    server = '0.0.0.0'
    port = '5000'
    app.run(host = server,  port = port, debug = True)
