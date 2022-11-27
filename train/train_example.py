from datasets import load_dataset

dataset_id = "embedding-data/QQP_triplets"

dataset = load_dataset(dataset_id)

print(f"- The {dataset_id} dataset has {dataset['train'].num_rows} examples.")
print(f"- Each example is a {type(dataset['train'][0])} with a {type(dataset['train'][0]['set'])} as value.")
print(f"- Examples look like this: {dataset['train'][0]}")

from sentence_transformers import InputExample

train_examples = []
train_data = dataset['train']['set']
n_examples = dataset['train'].num_rows // 100 # for performance, we only use 1/100 of the dataset

for i in range(n_examples):
  example = train_data[i]
  train_examples.append(InputExample(texts=[example['query'], example['pos'][0], example['neg'][0]]))

print(f"We have a {type(train_examples)} of length {len(train_examples)} containing {type(train_examples[0])}'s.")

from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

from sentence_transformers import losses

train_loss = losses.TripletLoss(model=model)

num_epochs = 10

warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) #10% of train data

model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=warmup_steps) 