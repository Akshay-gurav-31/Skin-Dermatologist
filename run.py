import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import pandas as pd
from flask import Flask, request, render_template, url_for
from io import BytesIO
from werkzeug.utils import secure_filename
import requests
import json
import base64

# ====================
# 1. Configuration
# ====================
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CHECKPOINT_PATH = 'checkpoints/Swin_MC_best_model.pth'
TRAIN_CSV = 'METADATA/Skin_Metadata.csv'
GEMINI_API_KEY = 'AIzaSyBJjVZ3US7q5hisidC3ToedcDx6gcfoFi4'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(os.path.join(app.root_path, UPLOAD_FOLDER), exist_ok=True)


# -----------------
# 2. Model Definition
# -----------------
class SwinClassifier(nn.Module):
    def __init__(self, num_classes: int, model_name: str = "swin_base_patch4_window12_384"):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
            global_pool="avg",
            img_size=512
        )

    def forward(self, x: torch.Tensor):
        return self.backbone(x)


# -----------------
# 3. Load Model & Metadata
# -----------------
def load_model_and_metadata():
    df = pd.read_csv(TRAIN_CSV)

    disease_labels = sorted(df['Disease_label'].dropna().unique().tolist())
    sub_labels = sorted(df['Sub_class'].dropna().unique().tolist())
    main_labels = sorted(df['Main_class'].dropna().unique().tolist())

    disease_map = {i: lbl for i, lbl in enumerate(disease_labels)}
    sub_map = {i: lbl for i, lbl in enumerate(sub_labels)}
    main_map = {i: lbl for i, lbl in enumerate(main_labels)}

    num_classes = len(main_labels)
    model = SwinClassifier(num_classes=num_classes).to(DEVICE)

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        new_state_dict = {}
        for k, v in checkpoint.items():
            if not k.startswith('backbone.'):
                k = 'backbone.' + k
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print(f"Warning: Checkpoint not found at {CHECKPOINT_PATH}")

    model.eval()

    mean = [0.53749797, 0.45875554, 0.40382471]
    std = [0.21629889, 0.20366619, 0.20136241]

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return model, disease_map, sub_map, main_map, transform


MODEL, DISEASE_MAP, SUB_MAP, MAIN_MAP, TRANSFORM = load_model_and_metadata()


# -----------------
# 4. Gemini API Function
# -----------------
def get_disease_info_from_gemini(disease_name, sub_class, main_class, image_base64, top3_str):
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

        prompt = f"""
        You are an expert dermatologist consultant. 
        
        **Context:**
        A machine learning model has analyzed the uploaded skin image.
        **Top 3 Probable Conditions:** {top3_str}
        
        **Primary Classification:**
        - **Disease Name:** {disease_name}
        - **Subclass:** {sub_class}
        - **Main Class:** {main_class}

        **Your Task:**
        1. Analyze the visual features of the skin condition in the image provided.
        2. Compare the visual evidence against the model's top predictions.
        3. Provide a comprehensive, professional, and empathetic report.
        4. **Format:** Return ONLY clean, semantic HTML (no markdown backticks, no <html>/<body> tags). Use Bootstrap 5 classes for styling.

        **Instructions for Analysis:**
        - Examine the skin lesion in the provided image carefully
        - Describe specific visual characteristics you observe (color, texture, shape, borders, size, etc.)
        - Explain how these features relate to the predicted conditions
        - Provide evidence-based recommendations for care
        - Do NOT use placeholders or bracketed text - provide specific, detailed information
        - Do NOT mention that this is AI-generated or from an API - present it as a professional dermatological analysis

        **Structure:**
        <div class="disease-report">
            <div class="alert alert-info mb-4 shadow-sm">
                <h4 class="alert-heading"><i class="bi bi-info-circle-fill me-2"></i>Model Analysis</h4>
                <p class="mb-0">Based on visual pattern analysis, <strong>{disease_name}</strong> has been identified as the most likely condition. Related conditions to consider include <em>{top3_str.replace(disease_name, '').replace('()', '').strip(', ')}</em>. <br><em>This is a model-based assessment for informational purposes only.</em></p>
            </div>

            <div class="row g-4">
                <div class="col-md-6">
                    <div class="card h-100 border-0 shadow-sm hover-effect">
                        <div class="card-body">
                            <h5 class="card-title text-primary"><i class="bi bi-activity me-2"></i>Visual Characteristics</h5>
                            <p class="card-text">Key visual features observed in this case:</p>
                            <ul>
                                <li>Describe specific visual features from the image, such as color variations, texture patterns, border definition, and surface characteristics</li>
                                <li>Explain how these features align with or differ from typical presentations of the predicted condition</li>
                            </ul>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card h-100 border-0 shadow-sm hover-effect">
                        <div class="card-body">
                            <h5 class="card-title text-success"><i class="bi bi-bandaid me-2"></i>Management Approach</h5>
                            <ul>
                                <li>Provide specific, evidence-based treatment options appropriate for this condition</li>
                                <li>Suggest practical home care measures that may help</li>
                                <li>Include any relevant lifestyle modifications</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>

            <div class="mt-4 p-3 bg-light rounded-3 border">
                <h5 class="text-danger"><i class="bi bi-exclamation-triangle-fill me-2"></i>Professional Consultation</h5>
                <p>Please consult a dermatologist promptly if you notice rapid changes in size, color, or shape, persistent bleeding, pain, or if the lesion becomes ulcerated. Early professional evaluation is crucial for accurate diagnosis and appropriate treatment.</p>
            </div>
            
            <div class="mt-3">
                 <h5 class="text-secondary"><i class="bi bi-shield-check me-2"></i>Prevention</h5>
                 <p>Practice sun protection with broad-spectrum SPF 30+ sunscreen, protective clothing, and shade-seeking behaviors. Maintain good skin hygiene and monitor for any new or changing lesions. Regular self-examinations can help detect early changes.</p>
            </div>
        </div>
        """

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_base64
                        }
                    }
                ]
            }],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 2000}
        }

        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                content = result['candidates'][0]['content']['parts'][0]['text']
                content = content.replace('```html', '').replace('```', '')
                return content

        return f"<div class='alert alert-warning'>Detailed analysis is currently unavailable (API Error: {response.status_code}). Please consult a doctor.</div>"
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return f"<div class='alert alert-danger'>An error occurred while fetching details: {str(e)}</div>"


# -----------------
# 5. Prediction Function
# -----------------
# Load valid skin disease labels
VALID_SKIN_DISEASES = set(pd.read_csv(TRAIN_CSV)['Disease_label'].dropna().unique().tolist())

def predict_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    img_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(img_tensor)
        probs = torch.softmax(logits, dim=1)

    # Get Top 3 Predictions
    top3_prob, top3_id = torch.topk(probs, 3, dim=1)
    
    top3_results = []
    for i in range(3):
        idx_i = int(top3_id[0][i].item())
        prob_i = top3_prob[0][i].item() * 100
        
        top3_results.append({
            'disease': DISEASE_MAP.get(idx_i, "Unknown"),
            'prob': f"{prob_i:.2f}",
            'width': f"{prob_i:.2f}%"
        })

    # Top 1 for main display
    idx = int(top3_id[0][0].item())
    predicted_prob = top3_prob[0][0].item()
    
    disease = DISEASE_MAP.get(idx, "Unknown Disease")
    sub_class = SUB_MAP.get(idx, "Unknown Subclass")
    main_class = MAIN_MAP.get(idx, "Unknown Mainclass")
    conf = f"{predicted_prob * 100:.2f}%"
    conf_value = predicted_prob * 100  # Numeric value for progress bar

    # Check if predicted disease is a valid skin condition
    if disease not in VALID_SKIN_DISEASES:
        # Not a valid skin condition, return appropriate message
        return {
            'disease': 'Not Found',
            'sub_class': 'Invalid Image',
            'main_class': 'Please upload a skin image',
            'conf_disease': '0%',
            'conf_value': 0,
            'top3': [],
            'disease_info': '<div class="alert alert-warning"><h4>Invalid Image</h4><p>The uploaded image does not appear to be a skin condition. Please upload a clear image of a skin area.</p></div>'
        }

    # Encode image for Gemini
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Pass top 3 to Gemini for context
    top3_str = ", ".join([f"{r['disease']} ({r['prob']}%)" for r in top3_results])
    disease_info = get_disease_info_from_gemini(disease, sub_class, main_class, image_base64, top3_str)

    return {
        'disease': disease,
        'sub_class': sub_class,
        'main_class': main_class,
        'conf_disease': conf,
        'conf_value': conf_value,
        'top3': top3_results,
        'disease_info': disease_info
    }


# -----------------
# 6. Flask Routes
# -----------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_url = None

    if request.method == 'POST':
        print("Received POST request")
        if 'file' in request.files:
            file = request.files['file']
            print(f"File received: {file.filename}")

            if file and file.filename and allowed_file(file.filename):
                image_bytes = file.read()
                print(f"Image bytes length: {len(image_bytes)}")

                # Save file first
                filename = secure_filename(file.filename)
                print(f"Secure filename: {filename}")
                
                img_to_save = Image.open(BytesIO(image_bytes))
                if img_to_save.mode == 'RGBA':
                    img_to_save = img_to_save.convert('RGB')

                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                img_to_save.save(path)
                image_url = url_for('static', filename=f'uploads/{filename}')
                print(f"Image saved to: {path}")
                print(f"Image URL: {image_url}")

                # Then predict
                try:
                    result = predict_image(image_bytes)
                    print(f"Prediction result: {result}")
                except Exception as e:
                    print(f"Prediction error: {e}")
                    # Return error information to the template
                    result = {'error': str(e)}

    return render_template('index.html', result=result, image_url=image_url)


# -----------------
# 7. Run App
# -----------------
if __name__ == "__main__":
    if 'MODEL' in globals() and MODEL is not None:
        app.run(debug=True, use_reloader=False)
    else:
        print("Model could not be loaded. Application stopped.")
