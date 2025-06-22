from flask import Flask, request, render_template, send_file
import torch
from torchvision import transforms
from PIL import Image
import os
from model.enhanced_generator import EnhancedGenerator, get_edge_map

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EnhancedGenerator().to(device)
model.load_state_dict(torch.load('model/generator_best.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

inv_transform = transforms.Compose([
    transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
    transforms.ToPILImage()
])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        img = Image.open(file).convert('RGB')
        x = transform(img)
        edge = get_edge_map(x)
        input_tensor = torch.cat([x, edge], dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = model(input_tensor).squeeze(0).cpu().clamp(-1, 1)

        output_tensor = (output_tensor + 1) / 2
        output_img = transforms.ToPILImage()(output_tensor)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.png')
        output_img.save(output_path)

        return render_template('index.html', output_image=output_path)
    return render_template('index.html')

@app.route('/download')
def download():
    return send_file("static/output.png", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
