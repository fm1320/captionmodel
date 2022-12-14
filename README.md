# Caption model for similarity

This is a small Flask API application with one API that serves a Pytorch model which calculates similarity between text and image captions.
The models used are from the hugging face library and use the [ViT-GPT2](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) transformer model for image captioning and [Sentence transformers](https://www.sbert.net/) 
for encoding the text sentences.
The image example is from the [Flickr8k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) dataset.
The two models are pre-trained and wrapped within a Pytorch module in order to enable training both models as a one Pytorch model.
A training example for the sentence transformer is also given for fine-tuning.

## Folder structure

```bash
.
├── ./artifacts/
│   └── ./artifacts/img_1.jpg
├── ./scripts/
│   ├── ./scripts/app.py
│   └── ./scripts/model.py
├── ./train/
│   └── ./train/train_example.py
└── ./requirements.txt
```

## How to run

Install [python](https://www.python.org/downloads/) and use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages.

```bash
$ pip install -r requirements.txt
```
Store the files in the proposed folder structure.
Navigate to the directory and run the ```model.py``` file.

```bash
$ python model.py
```
This should save a trained pytorch model ```model_0.1.0.pt``` in the directory. This will be used by the Flask app file later.
Then run the ```app.py``` file:

```bash
$ python app.py
```
After running this file you should see the flask server running
and see something along the lines of:
```
* Serving Flask app 'yes'
* Debug mode: off
* Running on http://127.0.0.1:5000
```
In order to use the flask API and send a request, a new terminal window should be opened.
The following command should be ran:

```bash
$curl -X POST http://127.0.0.1:5000/process_image -d "{\"image_name\":\"img_1.jpg\",\"text\":\"Car driving outside,A girl playing tennis,The dog running\"}" -H "Content-Type: application/json"
```

This curl command takes and image path and text strings as input.
The caption of the image file will be compared to each text example and the similarity will be calculated.

If you want to run the model with your own custom image and text, the following format of the curl command should be followed:

```bash
$curl -X POST http://127.0.0.1:5000/process_image -d "{\"image_name\":\"<path_to_image_file>\",\"text\":\"<Example text 1>,<Example text 2>,<Example text 3>\"}" -H "Content-Type: application/json"
```

* If double quotes are used for the curl command parameters, escape characters should be used for the double quotes inside the JSON data.


## Training

The flask API uses pre-trained models. The ```./train/train_example.py``` is an example on how to fine tune the sentence similarity model using the [QQP triplets](https://huggingface.co/datasets/embedding-data/QQP_triplets) dataset. However, in the case of the API service this was note used. This is just provided as an option for the user, if the relevant data is avaiable, the model can be fine-tuned to a specific task for better performance.


## License

[MIT](https://choosealicense.com/licenses/mit/)
