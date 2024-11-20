from flask import Flask, render_template, request
import pickle
import re
import os
import fitz  # PyMuPDF


app = Flask(__name__)

def cleanResume(resume_text):
    resume_text = re.sub('http\S+\s*', ' ', resume_text)  # remove URLs
    resume_text = re.sub('RT|cc', ' ', resume_text)  # remove RT and cc
    resume_text = re.sub('#\S+', '', resume_text)  # remove hashtags
    resume_text = re.sub('@\S+', '  ', resume_text)  # remove mentions
    resume_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resume_text)  # remove punctuations
    resume_text = re.sub(r'[^\x00-\x7f]',r' ', resume_text)
    resume_text = re.sub('\s+', ' ', resume_text)  # remove extra whitespace
    return resume_text

def extract_text_from_pdf(pdf_path):
    pdf_reader = fitz.open(pdf_path)
    text = ""
    for page_num in range(pdf_reader.page_count):
        page = pdf_reader[page_num]
        text += page.get_text()
    return text

def predict_category(clean_resume, model, tfidf_model, encoder_dict):
    vec = tfidf_model.transform([clean_resume])
    prediction = model.predict(vec)
    predicted_class = encoder_dict.inverse_transform(prediction)[0]
    return predicted_class

@app.route('/')
def index():
    return render_template('index.html')

from flask import render_template

@app.route('/resume_matcher', methods=['GET', 'POST'])
def resume_matcher():
    if request.method == 'POST':
        uploaded_files = request.files.getlist("file")
        try:
            model = pickle.load(open('ResumeBuilder.pkl', 'rb'))
            tfidf_model = pickle.load(open('tfidf_file.pkl', 'rb'))
            encoder_dict = pickle.load(open('label_encoder_file.pkl', 'rb'))
        except Exception as e:
            return f"Error loading model or files: {e}"

        result_messages = []
        os.makedirs("temp", exist_ok=True)
        
        for uploaded_file in uploaded_files:
            file_path = os.path.join("temp", uploaded_file.filename)
            uploaded_file.save(file_path)

            # Extract text from the PDF file
            resume_text = extract_text_from_pdf(file_path)

            # Clean the resume text
            clean_resume= cleanResume(resume_text)

            # Predict the category
            predicted_category = predict_category(clean_resume, model, tfidf_model, encoder_dict)

            result_messages.append(f"Resume in file {uploaded_file.filename} is classified under {predicted_category}")

            os.remove(file_path)

        return render_template('resume_matcher.html', result_messages=result_messages)

    return render_template('resume_matcher.html')


if __name__ == '__main__':
    app.run(debug=True)
