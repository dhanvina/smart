import os
from flask import Flask, render_template, request
from image_processings import divide_image
from mnist_model import process_and_predict_mnist
import shutil

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['GET','POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    # Save the uploaded image to a temporary folder
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Divide the uploaded image into cells
    divide_image(file_path)

    # Process the cell images and make predictions
    result = process_and_predict_mnist()
    shutil.rmtree('row')
    for filename in os.listdir('uploads'):
        file_path = os.path.join('uploads', filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    return f"Predicted Class Sums for Each Row: {result}, Total Sum: {sum(result)}"



if __name__ == '__main__':
    app.run(debug=True)
