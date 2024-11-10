import os
import gradio as gr
import docx2txt
import PyPDF2
import csv
from groq import Groq

# Ensure the environment variable is set correctly
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")  # Read from secrets, if set correctly

# Initialize the Groq client using the API key from the environment variable
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def extract_text_from_file(file):
    """
    Extracts text from a PDF or Word document file.
    """
    text = ""
    if file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file.name.endswith(".docx"):
        text = docx2txt.process(file.name)
    return text

def classify_and_generate_questions(text_chunk, question_type="MCQ"):
    """
    Uses Groq's Llama model to classify content and generate specific types of questions.
    """
    # Define prompt templates for each question type
    question_prompts = {
        "MCQ": f"Based on the following content, generate a multiple-choice question (MCQ) with four answer options. Specify the correct answer.\n\nText:\n{text_chunk}\n\nResponse:",
        "Fill-in-the-Blank": f"Based on the following content, create a fill-in-the-blank question by leaving out an important word or concept.\n\nText:\n{text_chunk}\n\nResponse:",
        "Short Answer": f"Based on the following content, generate a short answer question.\n\nText:\n{text_chunk}\n\nResponse:",
        "Concept Explanation": f"Based on the following content, create a question that requires explaining the concept in detail.\n\nText:\n{text_chunk}\n\nResponse:",
        "Numerical": f"Based on the following content, create a numerical question. Provide the correct answer if possible.\n\nText:\n{text_chunk}\n\nResponse:",
    }

    # Select the prompt based on the question type
    prompt = question_prompts.get(question_type, question_prompts["MCQ"])
    
    # Perform chat completion using Groq's Llama model
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama3-8b-8192"  # Replace with the model name you have access to
    )
    
    # Extract the response content
    return chat_completion.choices[0].message.content

def generate_questions(file, question_type, save_format):
    """
    Main function to process the file, generate questions, and optionally save them.
    """
    # Extract text
    text = extract_text_from_file(file)
    if not text:
        return "Could not extract text from the document.", None

    # Split text into chunks to handle model limits
    chunk_size = 500
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Generate questions for each chunk
    questions = []
    for chunk in text_chunks:
        question = classify_and_generate_questions(chunk, question_type=question_type)
        questions.append(question)

    # Format questions for display
    questions_display = "\n\n".join(questions)

    # Save questions if selected
    if save_format == "Text File":
        with open("generated_questions.txt", "w") as f:
            for question in questions:
                f.write(question + "\n\n")
        return questions_display, "generated_questions.txt"

    elif save_format == "CSV File":
        with open("generated_questions.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Question"])
            for question in questions:
                writer.writerow([question])
        return questions_display, "generated_questions.csv"

    return questions_display, None

# Gradio Interface
file_input = gr.File(label="Upload PDF or Word Document")
question_type = gr.Dropdown(["MCQ", "Fill-in-the-Blank", "Short Answer", "Concept Explanation", "Numerical"], label="Question Type")
save_format = gr.Dropdown(["None", "Text File", "CSV File"], label="Save Questions As")
output_text = gr.Textbox(label="Generated Questions")
output_file = gr.File(label="Download File")

gr.Interface(
    fn=generate_questions,
    inputs=[file_input, question_type, save_format],
    outputs=[output_text, output_file],
    title="Engineering Quiz Question Generator",
    description="Upload a PDF or Word document, choose the type of quiz question, and save them to a file if desired."
).launch()
