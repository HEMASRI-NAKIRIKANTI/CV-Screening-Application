from flask import Flask, render_template, request
import mammoth
import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAI
from flask_cors import CORS,cross_origin


nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__,static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'uploads'

# Set your OpenAI API key
import os
os.environ["OPENAI_API_KEY"] = 'sk-proj-d0aTu5c89wRMLAOk9WklT3BlbkFJ454ljJvjWZM39EPqV6W5'
# openai.api_key = 'sk-proj-d0aTu5c89wRMLAOk9WklT3BlbkFJ454ljJvjWZM39EPqV6W5'
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
        return text

def extract_text_from_docx(docx_file):
    result = mammoth.extract_raw_text(docx_file)
    return result.value


def extract_text_from_txt(file):
    # Ensure the UPLOAD_FOLDER exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # Save the uploaded file to the UPLOAD_FOLDER
    txt_file = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(txt_file)

    # Now, extract text from the saved file
    with open(txt_file, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Remove punctuation
    tokens = [word for word in tokens if word.isalnum()]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text
def calculate_similarity(cv_text_preprocessed, job_desc_text_preprocessed):
    # Combine preprocessed CV text and job description text into a single list
    documents = [cv_text_preprocessed, job_desc_text_preprocessed]

    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the documents and transform them into TF-IDF vectors
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    # Calculate cosine similarity between TF-IDF vectors
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Get the similarity score between the two documents
    similarity_score = similarity_matrix[0, 1]

    return similarity_score

@app.route('/',methods=['GET','POST'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET','POST'])
@cross_origin()
def upload():
    # Handle file uploads and process them
    cv_file = request.files['cv']
    job_desc_file = request.files['jobDescription']

    if not cv_file or not job_desc_file:
        return "Please upload both files"

    cv_text = ''
    job_desc_text = ''

    #extract text from cv based on doc type
    if cv_file.filename.endswith('.pdf'):
      cv_text = extract_text_from_pdf(cv_file)
    elif cv_file.filename.endswith('.docx'):
       cv_text = extract_text_from_docx(cv_file)
    elif cv_file.filename.endswith('.txt'):
       cv_text = extract_text_from_txt(cv_file)
    # print(cv_text)
    # extract text from Job Description based on doc type
    if job_desc_file.filename.endswith('.pdf'):
        job_desc_text = extract_text_from_pdf(job_desc_file)
    elif job_desc_file.filename.endswith('.docx'):
        job_desc_text = extract_text_from_docx(job_desc_file)
    elif job_desc_file.filename.endswith('.txt'):
        # Save the file to a temporary location
        job_desc_text = extract_text_from_txt(job_desc_file)
    print(job_desc_text)

    # Preprocess CV text
    cv_text_preprocessed = preprocess_text(cv_text)
    # Preprocess job description text
    job_desc_text_preprocessed = preprocess_text(job_desc_text)

    similarity_score = calculate_similarity(cv_text_preprocessed, job_desc_text_preprocessed)
    print("Similarity Score:", similarity_score)

    similarity_score = similarity_score*10

    # Prompt engineering: Combine CV and job description to form a prompt
    prompt = f"""CV: {cv_text}
    Job Description: {job_desc_text}
    Imagine you are a hiring manager tasked with evaluating a candidate for the role in the job description at a leading tech company. The candidate's CV highlights their expertise in skills in the resume, along with experience in developing scalable solutions. The job description emphasizes the importance of strategic thinking, problem-solving abilities, and collaboration skills. Your task is to thoroughly assess the candidate's qualifications, considering technical proficiency, project experience, and soft skills. Provide a detailed analysis of how well the candidate's background aligns with the requirements of the role, identifying strengths, potential areas for improvement, and overall suitability. Your evaluation should go beyond a simple numerical score, providing detailed insights and actionable recommendations for the hiring team.

    On a scale of 1 to 10, where 1 represents poor alignment and 10 represents perfect alignment, how would you rate the candidate's suitability for the role?
    """

    # Make a request to OpenAI API
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
    completion = llm.invoke(prompt)
    print(completion)

    return render_template('results.html', score=similarity_score, explanation=completion)

if __name__ == '__main__':
    app.run(debug=True)
