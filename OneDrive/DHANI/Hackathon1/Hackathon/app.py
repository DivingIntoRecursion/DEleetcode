from flask import Flask, render_template, request
from transformers import pipeline


from huggingface_hub import login

login('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')

# Initialize Flask App
app = Flask(__name__)

# Initialize Hugging Face Text Generation Pipeline
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')  # You can change this to another Hugging Face model

# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Donor Message API
@app.route('/donor', methods=['GET', 'POST'])
def donor_message():
    if request.method == 'POST':
        donor_name = request.form['donor_name']
        meals = request.form['meals']
        time = request.form['time']
        contact = request.form['contact']
        
        # Create prompt for Hugging Face model
        prompt = f"Draft a thank-you message for a donor named {donor_name} who is contributing {meals} meals for distribution. Include pickup time at {time}."
        response = generator(prompt, max_length=200, num_return_sequences=1)
        message = response[0]['generated_text']
        return render_template('donor.html', message=message)
    return render_template('donor.html', message=None)

# Volunteer Guide API
@app.route('/volunteer', methods=['GET', 'POST'])
def volunteer_guide():
    if request.method == 'POST':
        volunteer_name = request.form['volunteer_name']
        area = request.form['area']
        route = request.form['route']
        time = request.form['time']
        
        # Create prompt for Hugging Face model
        prompt = f"Create a step-by-step guide for a volunteer named {volunteer_name} to distribute food in {area} using route {route}. The distribution starts at {time}."
        response = generator(prompt, max_length=300, num_return_sequences=1)
        guide = response[0]['generated_text']
        return render_template('volunteer.html', guide=guide)
    return render_template('volunteer.html', guide=None)

# Food Distribution Plan API
@app.route('/distribution', methods=['GET', 'POST'])
def distribution_plan():
    if request.method == 'POST':
        areas = request.form['areas']
        meals = request.form['meals']
        demographics = request.form['demographics']
        
        # Create prompt for Hugging Face model
        prompt = f"Generate a food distribution plan for {meals} meals across these areas: {areas}. Prioritize based on demographics: {demographics}."
        response = generator(prompt, max_length=300, num_return_sequences=1)
        plan = response[0]['generated_text']
        return render_template('distribution.html', plan=plan)
    return render_template('distribution.html', plan=None)

if __name__ == '__main__':
    app.run(debug=True)
