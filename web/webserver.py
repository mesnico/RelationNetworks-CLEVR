import sys
sys.path.append("..")
from flask import Flask
from flask import render_template
from flask import request
from flask import send_from_directory
import utils
import evaluate
import os
import json

#TODO: put into configuration
clevr_dir = '/media/nicola/26F0A7D2064A1E46/TESI/CLEVR_v1.0'

qst_json = 'CLEVR_val_questions.json'
qst_json = os.path.join(clevr_dir, 'questions', qst_json)

#load question to image mapping
print('Loading images-questions mapping')
img_to_qst = utils.load_images_question_mappings(qst_json)

#initialize the evaluator (loads the model)
x = lambda: None #creates an empty object
x.no_cuda = True
x.resume = '../pretrained_models/original_fp_epoch_493.pth'
x.clevr_dir = clevr_dir
x.model = 'original-fp'
x.no_invert_questions = False
x.dropout = -1
x.config = '../config.json'
x.question_injection = -1
x.seed = 42

ev = evaluate.Evaluator(x)

#load question database #TODO: make shure database exists
qst_db = utils.JsonCache(qst_json, 'questions', all_in_memory=False)

app = Flask(__name__)

@app.route('/')
def initial_page():
    return render_template('index.html')


@app.route('/<string:page_name>')
def static_pages(page_name):
    return render_template('{}'.format(page_name))

@app.route('/clevr-imgs/<path:filename>')
def download_file(filename):
    return send_from_directory('clevr-imgs',
                               filename, as_attachment=True)

@app.route('/requests/questions', methods=['GET'])
def load_questions():
    imgid = request.args.get('imgid')
    imgid = int(imgid)
    qst_ids = img_to_qst[imgid]
    retrieved = [{'id': qst_db[x]['question_index'], 'sentence':qst_db[x]['question']} for x in qst_ids]
    return json.dumps(retrieved)

@app.route('/requests/answer', methods=['GET'])
def compute_answer():
    qst_id = request.args.get('qstid')
    qst_id = int(qst_id)
    return json.dumps(ev.evaluate(int(qst_id)))

    
