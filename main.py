import json
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, T5ForConditionalGeneration, BartForConditionalGeneration, AutoModelForSeq2SeqLM, GPT2TokenizerFast, T5TokenizerFast, MT5Tokenizer, MT5ForConditionalGeneration, BartTokenizerFast, BertTokenizer, AutoTokenizer
import evaluate
from datasets import Dataset, load_dataset


################ tool ################
class LabelSmoothingCrossEntropy(nn.Module):
    ''' Cross Entropy Loss with label smoothing '''
    def __init__(self, label_smooth=0.1, class_num=None):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num

    def forward(self, pred, target):
        ''' 
        Args:
            pred: prediction of model output    [N, M]
            target: ground truth of sampler [N]
        '''
        eps = 1e-12
        
        if self.label_smooth is not None:
            # cross entropy loss with label smoothing
            logprobs = F.log_softmax(pred, dim=1)	# softmax + log
            target = F.one_hot(target, self.class_num)	# 转换成one-hot
            
            # label smoothing
            # 实现 1
            # target = (1.0-self.label_smooth)*target + self.label_smooth/self.class_num 	
            # 实现 2
            # implement 2
            target = torch.clamp(target.float(), min=self.label_smooth/(self.class_num-1), max=1.0-self.label_smooth)
            loss = -1*torch.sum(target*logprobs, 1)
        
        else:
            # standard cross entropy loss
            loss = -1.*pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred+eps).sum(dim=1))

        return loss.mean()



################ dataset processing ################
def load_data(file_path):
    tokenizer = AutoTokenizer.from_pretrained('yuanzhoulvpi/gpt2_chinese')
    tokenizer.pad_token = '[PAD]'
    tokenizer.padding_side = 'left'  # 设置padding_side为left
    data = {'article': [], 'summary': []}
    samples_loaded = 0
    with open(file_path, 'r') as f:
        for line in f:
            data_point = json.loads(line)
            article = ''.join(data_point['article']).replace(" ", "")
            summary = data_point['summary'].replace(" ", "")
            # Apply filter to exclude empty articles or summaries
            if article.strip() and summary.strip():  # Exclude if either article or summary is empty after stripping whitespace
                # Check if the length of the concatenated input exceeds the maximum length
                max_length = 1000 - 5  # 5 is for "article: " and " summary: " tokens
                if len(tokenizer(article+summary)['input_ids']) <= max_length:
                    data['article'].append(article)
                    data['summary'].append(summary)
            samples_loaded += 1
    return data

dataset1_path = "./dataset/trainset.csv"
dataset2_path = "./dataset/testset1.csv"
dataset3_path = "./dataset/testset2.csv"
dataset4_path = "./dataset/testset3.csv"
dataset5_path = "./dataset/pretrain.csv"

data_files = {
    "trainset": dataset1_path,
    "testset1": dataset2_path,
    "testset2": dataset3_path,
    "testset3": dataset4_path,
    "pretrain": dataset5_path,
}
datasets = load_dataset("csv", data_files=data_files, delimiter=",", ignore_verifications=True)
trainset = datasets["trainset"]
testset1 = datasets["testset1"]
testset2 = datasets["testset2"]
testset3 = datasets["testset3"]
pretrain = datasets["pretrain"]

print("train number:", len(trainset))
print("test 1 number:", len(testset1))
print("test 2 number:", len(testset2))
print("test 3 number:", len(testset3))
print("pretrain number:", len(pretrain))
print("example:", trainset[0])


def evaluate_gpt_batch(model, tokenizer, data_encodings, batch_size=32):
    model.eval()
    rouge = evaluate.load("rouge")

    all_predictions = []
    all_references = []

    # Adjust for batch processing
    for i in tqdm(range(0, len(data_encodings['input_ids']), batch_size)):
        input_ids = torch.tensor(data_encodings['input_ids'][i:i+batch_size]).to(model.device)
        attention_mask = torch.tensor(data_encodings['attention_mask'][i:i+batch_size]).to(model.device)

        # Generate text in batches
        generated_text_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=1024,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Decode generated text for each item in the batch
        for j in range(len(input_ids)):
            gen_ids = generated_text_ids[j]
            input_length = len(input_ids[j])
            generated_text = tokenizer.decode(gen_ids[input_length:], skip_special_tokens=True)
            all_predictions.append(generated_text)

        # Append references for each item in the batch
        all_references.extend(data_encodings['labels'][i:i+batch_size])  # Ensure labels are provided as summaries in text form
    
    print("generated example:", all_predictions[0])
    # Compute metrics
    result = rouge.compute(predictions=all_predictions, references=all_references, rouge_types=["rouge1", "rouge2", "rougeL"])
    return {
        "rouge1": result["rouge1"]*100,
        "rouge2": result["rouge2"]*100,
        "rougeL": result["rougeL"]*100,
    }

def evaluate_model_batch(model, tokenizer, data_encodings, batch_size=32):
    model.eval()
    rouge = evaluate.load("rouge")

    all_predictions = []
    all_references = []

    # Adjust for batch processing
    for i in tqdm(range(0, len(data_encodings['input_ids']), batch_size)):
        input_ids = torch.tensor(data_encodings['input_ids'][i:i+batch_size]).to(model.device)
        attention_mask = torch.tensor(data_encodings['attention_mask'][i:i+batch_size]).to(model.device)

        # Generate text in batches
        generated_text_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=1024,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Decode generated text for each item in the batch
        for gen_ids in generated_text_ids:
            generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            all_predictions.append(generated_text)

        # Append references for each item in the batch
        all_references.extend(data_encodings['labels'][i:i+batch_size])  # Ensure labels are provided as summaries in text form
    
    print("generated example:", all_predictions[0])
    # Compute metrics
    result = rouge.compute(predictions=all_predictions, references=all_references, rouge_types=["rouge1", "rouge2", "rougeL"])
    return {
        "rouge1": result["rouge1"]*100,
        "rouge2": result["rouge2"]*100,
        "rougeL": result["rougeL"]*100,
    }


# ################ 训练GPT-2 ################ 
def train_gpt2():
    file_path1 = './results/gpt2_test1.json'
    file_path2 = './results/gpt2_test2.json'
    file_path3 = './results/gpt2_test3.json'

    if os.path.exists(file_path1) and os.path.exists(file_path2):
        with open(file_path1, 'r') as f:
            data = json.load(f)
        print("Test 1 metrics:", data)
        with open(file_path2, 'r') as f:
            data = json.load(f)
        print("Test 2 metrics:", data)
        with open(file_path3, 'r') as f:
            data = json.load(f)
        print("Test 3 metrics:", data)
        return

    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained('yuanzhoulvpi/gpt2_chinese')
    model = AutoModelForCausalLM.from_pretrained('yuanzhoulvpi/gpt2_chinese', pad_token_id=tokenizer.eos_token_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # 设置padding_side为left
    # Preprocess the data
    def preprocess_train_function(examples):
        max_length = tokenizer.model_max_length
        # Tokenize the concatenation of article and summary for the inputs
        tokenized_inputs = tokenizer(["article: " + a + " summary: " + s for a, s in zip(examples['article'], examples['summary'])],
                                    max_length=max_length,
                                    truncation=True,
                                    padding="max_length",  # Ensure all sequences are padded to the same length
                                    return_tensors="pt")  # Return PyTorch tensors

        return {
            'input_ids': tokenized_inputs.input_ids,
            'attention_mask': tokenized_inputs.attention_mask,
            'labels': tokenized_inputs.input_ids.clone()  # Simplified, adjust based on your actual needs
        }
    def preprocess_test_function(examples):
        # Tokenize each article in the batch, without including the summary in the tokenization
        max_length = tokenizer.model_max_length
        tokenized_inputs = tokenizer(["article: " + a for a in examples['article']],
                                    max_length=max_length,
                                    truncation=True,
                                    padding="max_length",  # Ensure all sequences are padded to the same length
                                    return_tensors="pt")  # Return PyTorch tensors
        return {
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'labels': examples['summary']
        }
    train_encodings = trainset.map(preprocess_train_function, batched=True)
    test1_encodings = testset1.map(preprocess_test_function, batched=True)
    test2_encodings = testset2.map(preprocess_test_function, batched=True)
    test3_encodings = testset3.map(preprocess_test_function, batched=True)
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=50,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=32,
        warmup_steps=1000,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=0,  
        save_total_limit=0, 
        learning_rate=1e-5,
        lr_scheduler_type='linear',
    )
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_encodings,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    trainer.train()
    # 评估模型
    metrics1 = evaluate_gpt_batch(model, tokenizer, test1_encodings)
    print("Test 1 metrics:", metrics1)
    metrics2 = evaluate_gpt_batch(model, tokenizer, test2_encodings)
    print("Test 2 metrics:", metrics2)
    metrics3 = evaluate_gpt_batch(model, tokenizer, test3_encodings)
    print("Test 3 metrics:", metrics2)
    del model
    torch.cuda.empty_cache()
    with open(file_path1, 'w') as f:
        json.dump(metrics1, f)
    with open(file_path2, 'w') as f:
        json.dump(metrics2, f)
    with open(file_path3, 'w') as f:
        json.dump(metrics3, f)

################ 训练BART ################ 
def train_bart():
    file_path1 = './results/bart_test1.json'
    file_path2 = './results/bart_test2.json'
    file_path3 = './results/bart_test3.json'

    if os.path.exists(file_path1) and os.path.exists(file_path2):
        with open(file_path1, 'r') as f:
            data = json.load(f)
        print("Test 1 metrics:", data)
        with open(file_path2, 'r') as f:
            data = json.load(f)
        print("Test 2 metrics:", data)
        with open(file_path3, 'r') as f:
            data = json.load(f)
        print("Test 3 metrics:", data)
        return

    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained('voidful/bart-base-chinese')
    model = BartForConditionalGeneration.from_pretrained('voidful/bart-base-chinese')
    tokenizer.pad_token = '[PAD]'
    # Preprocess the data
    def preprocess_train_function(examples):
        # Concatenate article and summary with a separator
        source_texts = ["article: " + a + " </s> summary: " for a, s in zip(examples['article'], examples['summary'])]
        # Tokenize the inputs
        tokenized_inputs = tokenizer(source_texts, max_length=tokenizer.model_max_length, truncation=True, padding="max_length", return_tensors="pt")
        # Adjust labels
        # For BART, we need to use tokenizer to encode the summary text as well, then set these as the labels
        labels = tokenizer(examples['summary'], max_length=tokenizer.model_max_length, truncation=True, padding="max_length", return_tensors="pt").input_ids
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    def preprocess_test_function(examples):
        # Tokenize each article
        source_texts = ["article: " + a + " </s> summary: " for a in examples['article']]
        tokenized_inputs = tokenizer(source_texts, max_length=tokenizer.model_max_length, truncation=True, padding="max_length", return_tensors="pt")
        tokenized_inputs['labels'] = examples['summary']
        return tokenized_inputs
    train_encodings = trainset.map(preprocess_train_function, batched=True)
    test1_encodings = testset1.map(preprocess_test_function, batched=True)
    test2_encodings = testset2.map(preprocess_test_function, batched=True)
    test3_encodings = testset3.map(preprocess_test_function, batched=True)
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=50,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=32,
        warmup_steps=1000,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=0,  
        save_total_limit=0, 
        learning_rate=1e-5,
        lr_scheduler_type='linear',
    )
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_encodings,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    trainer.train()
    # 评估模型
    metrics1 = evaluate_model_batch(model, tokenizer, test1_encodings)
    print("Test 1 metrics:", metrics1)
    metrics2 = evaluate_model_batch(model, tokenizer, test2_encodings)
    print("Test 2 metrics:", metrics2)
    metrics3 = evaluate_model_batch(model, tokenizer, test3_encodings)
    print("Test 3 metrics:", metrics2)
    del model
    torch.cuda.empty_cache()
    with open(file_path1, 'w') as f:
        json.dump(metrics1, f)
    with open(file_path2, 'w') as f:
        json.dump(metrics2, f)
    with open(file_path3, 'w') as f:
        json.dump(metrics3, f)

################ 训练Flan T5 Small ################ 
def train_flant5_small():
    file_path1 = './results/flant5_test1.json'
    file_path2 = './results/flant5_test2.json'
    file_path3 = './results/flant5_test3.json'

    if os.path.exists(file_path1) and os.path.exists(file_path2):
        with open(file_path1, 'r') as f:
            data = json.load(f)
        print("Test 1 metrics:", data)
        with open(file_path2, 'r') as f:
            data = json.load(f)
        print("Test 2 metrics:", data)
        with open(file_path3, 'r') as f:
            data = json.load(f)
        print("Test 3 metrics:", data)
        return

    # Load the tokenizer and the model
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    tokenizer.pad_token = tokenizer.eos_token
    # Preprocess the data
    def preprocess_train_function(examples):
        # Concatenate article and summary with a separator
        source_texts = ["article: " + a + " </s> summary: " for a, s in zip(examples['article'], examples['summary'])]
        tokenized_inputs = tokenizer(source_texts, max_length=tokenizer.model_max_length, truncation=True, padding="max_length", return_tensors="pt")
        target_texts = [s for a, s in zip(examples['article'], examples['summary'])]
        labels = tokenizer(target_texts, max_length=tokenizer.model_max_length, truncation=True, padding="max_length", return_tensors="pt")['input_ids']
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    def preprocess_test_function(examples):
        # Tokenize each article
        source_texts = ["article: " + a + " </s> summary: " for a in examples['article']]
        tokenized_inputs = tokenizer(source_texts, max_length=tokenizer.model_max_length, truncation=True, padding="max_length", return_tensors="pt")
        tokenized_inputs['labels'] = examples['summary']
        return tokenized_inputs
    train_encodings = trainset.map(preprocess_train_function, batched=True)
    test1_encodings = testset1.map(preprocess_test_function, batched=True)
    test2_encodings = testset2.map(preprocess_test_function, batched=True)
    test3_encodings = testset3.map(preprocess_test_function, batched=True)
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=50,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=32,
        warmup_steps=1000,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=0,  
        save_total_limit=0, 
        learning_rate=1e-5,
        lr_scheduler_type='linear',
    )
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_encodings,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    trainer.train()
    # 评估模型
    metrics1 = evaluate_model_batch(model, tokenizer, test1_encodings)
    print("Test 1 metrics:", metrics1)
    metrics2 = evaluate_model_batch(model, tokenizer, test2_encodings)
    print("Test 2 metrics:", metrics2)
    metrics3 = evaluate_model_batch(model, tokenizer, test3_encodings)
    print("Test 3 metrics:", metrics2)
    del model
    torch.cuda.empty_cache()
    with open(file_path1, 'w') as f:
        json.dump(metrics1, f)
    with open(file_path2, 'w') as f:
        json.dump(metrics2, f)
    with open(file_path3, 'w') as f:
        json.dump(metrics3, f)

################ 训练Flan T5 Base ################ 
def train_flant5_base():
    file_path1 = './results/flant5_test1.json'
    file_path2 = './results/flant5_test2.json'
    file_path3 = './results/flant5_test3.json'

    if os.path.exists(file_path1) and os.path.exists(file_path2):
        with open(file_path1, 'r') as f:
            data = json.load(f)
        print("Test 1 metrics:", data)
        with open(file_path2, 'r') as f:
            data = json.load(f)
        print("Test 2 metrics:", data)
        with open(file_path3, 'r') as f:
            data = json.load(f)
        print("Test 3 metrics:", data)
        return

    # Load the tokenizer and the model
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    tokenizer.pad_token = tokenizer.eos_token
    # Preprocess the data
    def preprocess_train_function(examples):
        # Concatenate article and summary with a separator
        source_texts = ["article: " + a + " </s> summary: " for a, s in zip(examples['article'], examples['summary'])]
        tokenized_inputs = tokenizer(source_texts, max_length=tokenizer.model_max_length, truncation=True, padding="max_length", return_tensors="pt")
        target_texts = [s for a, s in zip(examples['article'], examples['summary'])]
        labels = tokenizer(target_texts, max_length=tokenizer.model_max_length, truncation=True, padding="max_length", return_tensors="pt")['input_ids']
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    def preprocess_test_function(examples):
        # Tokenize each article
        source_texts = ["article: " + a + " </s> summary: " for a in examples['article']]
        tokenized_inputs = tokenizer(source_texts, max_length=tokenizer.model_max_length, truncation=True, padding="max_length", return_tensors="pt")
        tokenized_inputs['labels'] = examples['summary']
        return tokenized_inputs
    train_encodings = trainset.map(preprocess_train_function, batched=True)
    test1_encodings = testset1.map(preprocess_test_function, batched=True)
    test2_encodings = testset2.map(preprocess_test_function, batched=True)
    test3_encodings = testset3.map(preprocess_test_function, batched=True)
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=50,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=32,
        warmup_steps=1000,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=0,  
        save_total_limit=0, 
        learning_rate=1e-5,
        lr_scheduler_type='linear',
    )
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_encodings,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    trainer.train()
    # 评估模型
    metrics1 = evaluate_model_batch(model, tokenizer, test1_encodings)
    print("Test 1 metrics:", metrics1)
    metrics2 = evaluate_model_batch(model, tokenizer, test2_encodings)
    print("Test 2 metrics:", metrics2)
    metrics3 = evaluate_model_batch(model, tokenizer, test3_encodings)
    print("Test 3 metrics:", metrics2)
    del model
    torch.cuda.empty_cache()
    with open(file_path1, 'w') as f:
        json.dump(metrics1, f)
    with open(file_path2, 'w') as f:
        json.dump(metrics2, f)
    with open(file_path3, 'w') as f:
        json.dump(metrics3, f)

################ 训练BART pre-training ################ 
def train_bart_pt():
    file_path1 = './results/bart_pt_test1.json'
    file_path2 = './results/bart_pt_test2.json'
    file_path3 = './results/bart_pt_test3.json'

    if os.path.exists(file_path1) and os.path.exists(file_path2):
        with open(file_path1, 'r') as f:
            data = json.load(f)
        print("Test 1 metrics:", data)
        with open(file_path2, 'r') as f:
            data = json.load(f)
        print("Test 2 metrics:", data)
        with open(file_path3, 'r') as f:
            data = json.load(f)
        print("Test 3 metrics:", data)
        return

    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained('voidful/bart-base-chinese')
    model = BartForConditionalGeneration.from_pretrained('voidful/bart-base-chinese')
    tokenizer.pad_token = '[PAD]'
    # Preprocess the data
    def preprocess_train_function(examples):
        # Concatenate article and summary with a separator
        source_texts = ["article: " + a + " </s> summary: " for a, s in zip(examples['article'], examples['summary'])]
        # Tokenize the inputs
        tokenized_inputs = tokenizer(source_texts, max_length=tokenizer.model_max_length, truncation=True, padding="max_length", return_tensors="pt")
        # Adjust labels
        # For BART, we need to use tokenizer to encode the summary text as well, then set these as the labels
        labels = tokenizer(examples['summary'], max_length=tokenizer.model_max_length, truncation=True, padding="max_length", return_tensors="pt").input_ids
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    def preprocess_test_function(examples):
        # Tokenize each article
        source_texts = ["article: " + a + " </s> summary: " for a in examples['article']]
        tokenized_inputs = tokenizer(source_texts, max_length=tokenizer.model_max_length, truncation=True, padding="max_length", return_tensors="pt")
        tokenized_inputs['labels'] = examples['summary']
        return tokenized_inputs
    pretrain_encodings = pretrain.map(preprocess_train_function, batched=True)
    train_encodings = trainset.map(preprocess_train_function, batched=True)
    test1_encodings = testset1.map(preprocess_test_function, batched=True)
    test2_encodings = testset2.map(preprocess_test_function, batched=True)
    test3_encodings = testset3.map(preprocess_test_function, batched=True)
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=500,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=32,
        warmup_steps=1000,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=0,  
        save_total_limit=0, 
        learning_rate=2e-5,
        lr_scheduler_type='linear',
    )
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=pretrain_encodings,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    trainer.train()
    # Initialize the Trainer
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=50,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=32,
        warmup_steps=1000,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=0,  
        save_total_limit=0, 
        learning_rate=1e-5,
        lr_scheduler_type='linear',
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_encodings,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    trainer.train()
    # 评估模型
    metrics1 = evaluate_model_batch(model, tokenizer, test1_encodings)
    print("Test 1 metrics:", metrics1)
    metrics2 = evaluate_model_batch(model, tokenizer, test2_encodings)
    print("Test 2 metrics:", metrics2)
    metrics3 = evaluate_model_batch(model, tokenizer, test3_encodings)
    print("Test 3 metrics:", metrics2)
    del model
    torch.cuda.empty_cache()
    with open(file_path1, 'w') as f:
        json.dump(metrics1, f)
    with open(file_path2, 'w') as f:
        json.dump(metrics2, f)
    with open(file_path3, 'w') as f:
        json.dump(metrics3, f)

################ 训练BART pre-training ################ 
def train_bart_LSR_pt():
    file_path1 = './results/bart_lsr_pt_test1.json'
    file_path2 = './results/bart_lsr_pt_test2.json'
    file_path3 = './results/bart_lsr_pt_test3.json'

    if os.path.exists(file_path1) and os.path.exists(file_path2):
        with open(file_path1, 'r') as f:
            data = json.load(f)
        print("Test 1 metrics:", data)
        with open(file_path2, 'r') as f:
            data = json.load(f)
        print("Test 2 metrics:", data)
        with open(file_path3, 'r') as f:
            data = json.load(f)
        print("Test 3 metrics:", data)
        return

    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained('voidful/bart-base-chinese')
    model = BartForConditionalGeneration.from_pretrained('voidful/bart-base-chinese')
    tokenizer.pad_token = '[PAD]'
    # Preprocess the data
    def preprocess_train_function(examples):
        # Concatenate article and summary with a separator
        source_texts = ["article: " + a + " </s> summary: " for a, s in zip(examples['article'], examples['summary'])]
        # Tokenize the inputs
        tokenized_inputs = tokenizer(source_texts, max_length=tokenizer.model_max_length, truncation=True, padding="max_length", return_tensors="pt")
        # Adjust labels
        # For BART, we need to use tokenizer to encode the summary text as well, then set these as the labels
        labels = tokenizer(examples['summary'], max_length=tokenizer.model_max_length, truncation=True, padding="max_length", return_tensors="pt").input_ids
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    def preprocess_test_function(examples):
        # Tokenize each article
        source_texts = ["article: " + a + " </s> summary: " for a in examples['article']]
        tokenized_inputs = tokenizer(source_texts, max_length=tokenizer.model_max_length, truncation=True, padding="max_length", return_tensors="pt")
        tokenized_inputs['labels'] = examples['summary']
        return tokenized_inputs
    pretrain_encodings = pretrain.map(preprocess_train_function, batched=True)
    train_encodings = trainset.map(preprocess_train_function, batched=True)
    test1_encodings = testset1.map(preprocess_test_function, batched=True)
    test2_encodings = testset2.map(preprocess_test_function, batched=True)
    test3_encodings = testset3.map(preprocess_test_function, batched=True)
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=500,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=32,
        warmup_steps=1000,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=0,  
        save_total_limit=0, 
        learning_rate=2e-5,
        lr_scheduler_type='linear',
    )

    # pretraining
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=pretrain_encodings,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    trainer.train()

    # finetuning
    class CustomTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.criterion = LabelSmoothingCrossEntropy(label_smooth=0.4, class_num=tokenizer.vocab_size)

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get('logits')
            loss = self.criterion(logits, labels)
            return (loss, outputs) if return_outputs else loss
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=50,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=32,
        warmup_steps=1000,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=0,  
        save_total_limit=0, 
        learning_rate=1e-5,
        lr_scheduler_type='linear',
    )
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_encodings,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    trainer.train()

    # 评估模型
    metrics1 = evaluate_model_batch(model, tokenizer, test1_encodings)
    print("Test 1 metrics:", metrics1)
    metrics2 = evaluate_model_batch(model, tokenizer, test2_encodings)
    print("Test 2 metrics:", metrics2)
    metrics3 = evaluate_model_batch(model, tokenizer, test3_encodings)
    print("Test 3 metrics:", metrics2)
    del model
    torch.cuda.empty_cache()
    with open(file_path1, 'w') as f:
        json.dump(metrics1, f)
    with open(file_path2, 'w') as f:
        json.dump(metrics2, f)
    with open(file_path3, 'w') as f:
        json.dump(metrics3, f)

train_gpt2()
train_bart()
train_flant5_small()
train_flant5_base()
train_bart_LSR_pt()
