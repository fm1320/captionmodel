import torch
from flask import Flask
from flask import request
from model import Task_Model, cos_sim, process_image, BASE_DIR, PATH
app = Flask('yes')
 
model_obj = Task_Model()
model_obj.load_state_dict(torch.load(PATH))
#model_obj.eval()
 
@app.route('/process_image', methods=['POST'])
def similiarty_score_for_image():
    data=request.get_json()
    image_name = data['image_name']
    image_path= [f"{BASE_DIR}/{image_name}"]
    text = data['text'].split(',')
    images1= process_image(image_path) 
 
    similarity_score = model_obj.forward(images1,text)
    return [float(similarity_score[1]),float(similarity_score[2]),float(similarity_score[3])],200 # had to return floats
 
app.run(host='127.0.0.1', port = 5000)