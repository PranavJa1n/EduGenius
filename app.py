import pymupdf
import os
from openai import OpenAI
from dotenv import load_dotenv
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from flask import Flask, render_template, request, redirect, url_for, send_file
import base64

load_dotenv()
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
GENERATED_FOLDER = 'generated'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf_file' not in request.files:
        return render_template('index.html', error="No file selected")
    
    file = request.files['pdf_file']
    if file.filename == '':
        return render_template('index.html', error="No file selected")
    
    if file and file.filename.lower().endswith('.pdf'):
        # Get user-defined limits
        mcq_limit = int(request.form.get('mcq_limit', 10))
        tf_limit = int(request.form.get('tf_limit', 10))
        
        # Save uploaded file
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the PDF and generate questions
            process_pdf_and_generate_questions(filepath, mcq_limit, tf_limit)
            
            # Convert PDFs to base64 for embedding
            question_pdf_base64 = pdf_to_base64('generated/question.pdf')
            answer_pdf_base64 = pdf_to_base64('generated/answer.pdf')
            
            return render_template('result.html', 
                                 question_pdf=question_pdf_base64,
                                 answer_pdf=answer_pdf_base64,
                                 mcq_count=mcq_limit,
                                 tf_count=tf_limit)
        
        except Exception as e:
            return render_template('index.html', error=f"Error processing PDF: {str(e)}")
    
    else:
        return render_template('index.html', error="Please upload a valid PDF file")

def process_pdf_and_generate_questions(filepath, mcq_limit, tf_limit):
    # Extract text from PDF
    with pymupdf.open(filepath) as doc:
        content = chr(12).join([page.get_text() for page in doc])
    
    content += f"\nNumber of MCQ = {mcq_limit}\nNumber of True/False = {tf_limit}"
    
    # Generate questions using OpenAI
    client = OpenAI(api_key=os.getenv("api"), base_url="https://api.perplexity.ai")
    response = client.chat.completions.create(
        model="sonar-pro",
        temperature=1.7,
        messages=[
            {
                "role": "user",
                "content": '''
Educational Assessment Generator
Role Definition
You are an experienced educator and assessment specialist tasked with creating comprehensive evaluations based on provided learning materials.
Primary Task
Generate multiple-choice questions (MCQs) and true/false questions exclusively from the content provided by the user. Do not incorporate external knowledge or information not present in the source material.
Default Parameters
* Quantity: 10 MCQs and 10 True/False questions (unless otherwise specified)
* Difficulty Level: Moderate - balanced between accessibility and challenge, avoiding overly simple or excessively complex questions
* Source Constraint: All questions must derive directly from the provided content only
Question Construction Guidelines
Multiple Choice Questions (MCQs)
* Create 4 answer options (A, B, C, D) for each question
* Ensure one clearly correct answer and three plausible distractors
* Vary question types: factual recall, comprehension, application, and analysis
* Use clear, unambiguous language
* Avoid "all of the above" or "none of the above" options unless strategically necessary
True/False Questions
* Create definitive statements that are clearly true or false based on the content
* Avoid ambiguous or partially correct statements
* Include a mix of straightforward facts and conceptual understanding
* Ensure statements test meaningful comprehension, not trivial details
Output Format Requirements
Structure your response as follows:
MULTIPLE CHOICE QUESTIONS
1. [Question text] A) [Option A] B) [Option B] C) [Option C] D) [Option D]
[Continue for all MCQs]
TRUE/FALSE QUESTIONS
1. [Statement]
2. [Statement]
[Continue for all T/F questions]
ANSWER KEY
Multiple Choice Answers:

1. [Correct letter]
2. [Correct letter] [Continue for all MCQs]

True/False Answers:

1. [True/False]
2. [True/False] [Continue for all T/F questions]
Quality Assurance
* Verify all questions can be answered using only the provided content
* Ensure moderate difficulty level throughout
* Check for clear, grammatically correct language
* Confirm answer key accuracy
Instructions for Use
Provide your content material, and specify any custom requirements for:
* Number of questions (if different from default 10 each)
* Difficulty level adjustment (if different from moderate)
* Specific focus areas within the content
Now please provide the content you'd like me to use for creating your assessment questions.
'''
            },
            {
                "role": "assistant",
                "content": '''
I understand you'd like me to create educational assessments based on learning materials you provide. I'm ready to generate comprehensive evaluations following your specifications:

**My Assessment Creation Capabilities:**
- **Default Output**: 10 multiple-choice questions (4 options each) + 10 true/false questions
- **Difficulty**: Moderate level balancing accessibility with challenge
- **Content Fidelity**: Questions derived exclusively from your provided materials
- **Question Types**: Mix of recall, comprehension, application, and analysis
- **Clear Formatting**: Organized with complete answer keys

**What I Need From You:**
Please share the learning content/materials you'd like me to use as the foundation for the assessment. This could be:
- Text passages or articles
- Lecture notes or transcripts
- Educational documents
- Course materials
- Any other written content

**Optional Customizations:**
You can also specify if you want:
- Different quantities (e.g., 15 MCQs, 5 T/F)
- Adjusted difficulty level (easier/harder)
- Focus on specific topics within your content
- Any particular question types or formats

Once you provide the source material, I'll create a comprehensive assessment that tests meaningful understanding of that specific content, complete with a detailed answer key.

What learning materials would you like me to work with?
'''
            },
            {
                "role": "user",
                "content": content
            }
        ],
        max_tokens=4000
    )
    
    # Process the response and create PDFs
    create_question_and_answer_pdfs(response.choices[0].message.content)

def wrap_text(text, font_name, font_size, max_width):
    """Wrap text to fit within max_width"""
    from reportlab.pdfbase.pdfmetrics import stringWidth
    
    if not text.strip():
        return [text]
    
    words = text.split(' ')
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        line_width = stringWidth(test_line, font_name, font_size)
        
        if line_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
                current_line = word
            else:
                # Single word is too long, just add it
                lines.append(word)
    
    if current_line:
        lines.append(current_line)
    
    return lines if lines else [text]

def clean_mathematical_text(text):
    """Clean and fix mathematical notation that got corrupted"""
    # Common replacements for corrupted mathematical symbols
    replacements = {
        # Fix escaped parentheses
        r'\(': '',
        r'\)': '',
        r'\\(': '',
        r'\\)': '',
        # Fix Greek letters
        r'\\mu': 'μ',
        r'\mu': 'μ',
        r'\\lambda': 'λ',
        r'\lambda': 'λ',
        r'\\Delta': 'Δ',
        r'\Delta': 'Δ',
        r'\\theta': 'θ',
        r'\theta': 'θ',
        r'\\pi': 'π',
        r'\pi': 'π',
        r'\\alpha': 'α',
        r'\alpha': 'α',
        r'\\beta': 'β',
        r'\beta': 'β',
        r'\\gamma': 'γ',
        r'\gamma': 'γ',
        # Fix common mathematical expressions
        r'2t = n\\lambda': '2t = nλ',
        r'2t = (2n-1)\\lambda/2': '2t = (2n-1)λ/2',
        r'2t = (n+1)\\lambda': '2t = (n+1)λ',
        r'2t = n\\lambda/2': '2t = nλ/2',
        # Fix superscripts - convert ^2 to ²
        r'\^2': '²',
        r'^2': '²',
        r'\^3': '³',
        r'^3': '³',
        # Fix square root
        r'\\sqrt\{': '√(',
        r'\sqrt\{': '√(',
        r'\\sqrt': '√',
        r'\sqrt': '√',
        # Fix subscripts and superscripts notation
        r'D_\{n,air\}\^2': 'D(n,air)²',
        r'D_\{n,liquid\}\^2': 'D(n,liquid)²',
        r'D_\{n,air\}': 'D(n,air)',
        r'D_\{n,liquid\}': 'D(n,liquid)',
        # Fix specific patterns from your example
        r'Dn = \\sqrt\{4Rnλ\}': 'Dn = √(4Rnλ)',
        r'Dn\^2 = 4Rnλ': 'Dn² = 4Rnλ',
        r'Dn\^2 = 2Rnλ': 'Dn² = 2Rnλ',
        r'Dn\^2 = Rnλ': 'Dn² = Rnλ',
        # Clean up extra backslashes
        r'\\': '',
    }
    
    result = text
    for old, new in replacements.items():
        result = result.replace(old, new)
    
    # Additional cleanup for any remaining LaTeX-style formatting
    import re
    # Remove any remaining \( and \) patterns
    result = re.sub(r'\\?\(', '', result)
    result = re.sub(r'\\?\)', '', result)
    # Fix any remaining curly braces
    result = result.replace('{', '(').replace('}', ')')
    
    return result

def create_question_and_answer_pdfs(content):
    # Keep the original content and clean mathematical notation
    cleaned_content = clean_mathematical_text(content)
    
    # More robust splitting for answer key
    answer_key_split = cleaned_content.split('ANSWER KEY')
    if len(answer_key_split) > 1:
        questions_part = answer_key_split[0].strip()
        answer_key_part = 'ANSWER KEY' + answer_key_split[1]
    else:
        questions_part = cleaned_content
        answer_key_part = ""
    
    # Split questions into MCQ and True/False
    mcq_split = questions_part.split('TRUE/FALSE QUESTIONS')
    mcq_section = mcq_split[0].strip()
    tf_section = mcq_split[1].strip() if len(mcq_split) > 1 else ""
    
    # Remove duplicate headings
    if mcq_section.startswith("MULTIPLE CHOICE QUESTIONS"):
        mcq_section = "\n".join(mcq_section.split("\n")[1:]).strip()
    
    if tf_section.startswith("TRUE/FALSE QUESTIONS"):
        tf_section = "\n".join(tf_section.split("\n")[1:]).strip()
    
    # Create question PDF
    c = canvas.Canvas("generated/question.pdf", pagesize=A4)
    width, height = A4
    
    # Set margins - left margin 40, right margin 40
    left_margin = 40
    right_margin = 40
    max_width = width - left_margin - right_margin
    
    # MCQ section
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(width/2, height - 40, "MULTIPLE CHOICE QUESTIONS")
    
    textobject = c.beginText(left_margin, height - 80)
    textobject.setFont("Helvetica", 12)
    line_height = 14
    margin_bottom = 80
    y = height - 80
    
    for line in mcq_section.split('\n'):
        # Clean mathematical notation
        display_line = clean_mathematical_text(line)
        
        # Wrap each line to fit within margins
        wrapped_lines = wrap_text(display_line, "Helvetica", 12, max_width)
        for wrapped_line in wrapped_lines:
            if y <= margin_bottom:
                c.drawText(textobject)
                c.showPage()
                textobject = c.beginText(left_margin, height - 80)
                textobject.setFont("Helvetica", 12)
                y = height - 80
            try:
                textobject.textLine(wrapped_line)
            except:
                # Fallback for problematic characters
                safe_line = wrapped_line.encode('ascii', 'replace').decode('ascii')
                textobject.textLine(safe_line)
            y -= line_height
    
    c.drawText(textobject)
    c.showPage()
    
    # True/False section
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(width/2, height - 40, "TRUE/FALSE QUESTIONS")
    textobject = c.beginText(left_margin, height - 80)
    textobject.setFont("Helvetica", 12)
    y = height - 80
    
    for line in tf_section.split('\n'):
        # Clean mathematical notation
        display_line = clean_mathematical_text(line)
        
        # Wrap each line to fit within margins
        wrapped_lines = wrap_text(display_line, "Helvetica", 12, max_width)
        for wrapped_line in wrapped_lines:
            if y <= margin_bottom:
                c.drawText(textobject)
                c.showPage()
                textobject = c.beginText(left_margin, height - 80)
                textobject.setFont("Helvetica", 12)
                y = height - 80
            try:
                textobject.textLine(wrapped_line)
            except:
                # Fallback for problematic characters
                safe_line = wrapped_line.encode('ascii', 'replace').decode('ascii')
                textobject.textLine(safe_line)
            y -= line_height
    
    c.drawText(textobject)
    c.save()
    
    # Create answer PDF
    c2 = canvas.Canvas("generated/answer.pdf", pagesize=A4)
    c2.setFont("Helvetica-Bold", 14)
    c2.drawCentredString(width/2, height - 40, "ANSWER KEY")
    
    textobject = c2.beginText(left_margin, height - 80)
    textobject.setFont("Helvetica", 12)
    y = height - 80
    
    # Only process answer key content, skip the "ANSWER KEY" header
    answer_lines = answer_key_part.split('\n')[1:] if answer_key_part else []
    
    for line in answer_lines:
        # Clean mathematical notation
        display_line = clean_mathematical_text(line)
        
        # Wrap each line to fit within margins
        wrapped_lines = wrap_text(display_line, "Helvetica", 12, max_width)
        for wrapped_line in wrapped_lines:
            if y <= margin_bottom:
                c2.drawText(textobject)
                c2.showPage()
                textobject = c2.beginText(left_margin, height - 80)
                textobject.setFont("Helvetica", 12)
                y = height - 80
            try:
                textobject.textLine(wrapped_line)
            except:
                # Fallback for problematic characters
                safe_line = wrapped_line.encode('ascii', 'replace').decode('ascii')
                textobject.textLine(safe_line)
            y -= line_height
    
    c2.drawText(textobject)
    c2.save()

def pdf_to_base64(filepath):
    """Convert PDF file to base64 string for embedding in HTML"""
    with open(filepath, 'rb') as pdf_file:
        pdf_data = pdf_file.read()
        return base64.b64encode(pdf_data).decode('utf-8')

@app.route('/download/<pdf_type>')
def download_pdf(pdf_type):
    """Route to download generated PDFs"""
    if pdf_type == 'questions':
        return send_file('generated/question.pdf', as_attachment=True, download_name='questions.pdf')
    elif pdf_type == 'answers':
        return send_file('generated/answer.pdf', as_attachment=True, download_name='answers.pdf')
    else:
        return redirect(url_for('home'))

@app.route('/help')
def help_page():
    return render_template("help.html")

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=80)