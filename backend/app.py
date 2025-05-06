from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)
CORS(app)

# Load HuggingFace model
print("Loading CodeGen model...")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-multi")
print("Model loaded.")

def generate_code(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, top_p=0.95, temperature=0.8)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated

@app.route('/generate_code', methods=['POST'])
def generate_code_api():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400
    code = generate_code(prompt)
    return jsonify({'code': code})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
