# Caption model for similarity

This is a flask API that serves a pytoch model which calculates similarity between text and image captions.
The models used are from the hugging face library and use the ViT transformer model for image captioning and Sentence transformer 
for encoding the text sentences.
The image example is from the Flickr8k dataset.
The two models are pre-trained and wrapped within a pytorch module in order to enable training both models as a one pytorch model.
A training example for the sentence transformer is also given for fine-tuning.

## Folder structure

```bash
.
├── ./model/
│   ├── ./model/main.py
│   ├── ./model/model.py
│   ├── ./model/model_0.1.0.pt
│   ├── ./model/requirements.txt
│   └── ./model/img_1.jpg
└── ./train/
    └── ./train/train_example.py
```

## How to run

Install python and use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages.

```bash
$ pip install -r requirements.txt
```
Store the files in the proposed folder structure.
Navigate to the directory and run the ```main.py``` file.

```bash
$ python run main.py
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

* If dobule quotes are used for the curl command parameters, escape charaacters should be also used for the double quotes inside the JSON file


## Training

A pre-trained pytorch model is also part of the repo ```model_0.1.0.pt```. The ```./train/train_example.py``` is an example on how to fine tune the sentence similarity model using the [QQP triplets](https://huggingface.co/datasets/embedding-data/QQP_triplets) dataset. However, in the case of the API service this was note used. This is just provided as an option for the user, if the relevant data is avaiable, the model can be fine-tuned to a specific task for better performance.


## License

[MIT](https://choosealicense.com/licenses/mit/)
