from tqdm.auto import tqdm
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import torch
import utils
import datasets
import evaluate
import numpy as np
import nltk
from torch.utils.data import DataLoader
import argparse






def train(num_datapoints=None):
    # Define the paths to the pre-trained model and tokenizer
    model_path = "models/finetuned"

    # Load the BART model and tokenizer
    tokenizer, model = utils.load_model()

    # Prepare the dataset
    dataset = utils.load_dataset(utils.DATASET_PATH)
    if num_datapoints:
        dataset = dataset.select(range(num_datapoints))

    dataset = dataset.train_test_split(test_size=0.2, seed=69)

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    train_dataset = utils.preprocess_dataset(train_dataset, tokenizer)
    eval_dataset = utils.preprocess_dataset(eval_dataset, tokenizer)

    
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_path,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,
        predict_with_generate=True,
        eval_accumulation_steps=3,
        fp16=True,  
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    metric = datasets.load_metric("rouge")
    nltk.download("punkt")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [
            "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
        ]
        decoded_labels = [
            "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
        ]

        result = metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        
        result = {key: value.mid.recall * 100 for key, value in result.items()}

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
        ]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}


    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.add_callback(utils.SummariesCallback(model, tokenizer))

    trainer.train()

    trainer.save_model(model_path)

    # Evaluate model
    metric = evaluate.load("rouge")
    test_dataloader = DataLoader(eval_dataset, batch_size=16, num_workers=4)
    for batch in tqdm(test_dataloader):
        references = batch["summary"]
        references = ["\n".join(nltk.sent_tokenize(ref.strip())) for ref in references]
        inputs = torch.stack(batch["input_ids"]).to(model.device).T
        masks = torch.stack(batch["attention_mask"]).to(model.device).T
        predictions = model.generate(inputs, attention_mask=masks, num_beams=4, max_length=128)
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        predictions = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in predictions]
        metric.add_batch(predictions=predictions, references=references)

    # Print the final results
    print(metric.compute())


    # Do one random example
    content = eval_dataset[np.random.randint(0, len(eval_dataset))]["content"]
    print("Below is an example text and model generated summary from the eval dataset")
    print("-"*15)
    print("Text:\n")
    print(content, "\n")
    print("Summary:\n")
    print(utils.generate_summary(model, tokenizer, content))
    print("-"*15)

    user_interface(model, tokenizer)

def user_interface(model, tokenizer):
    while True:
        text = input("Write a text and get a summary (q to quit): ")
        if text == "q":
            break
        print("\nSummary:\n")
        print(utils.generate_summary(model, tokenizer, text))
        print("-"*15)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')

    # Add arguments to the parser
    parser.add_argument('-l', '--load', type=str, help='If set, loads model from specified path')
    parser.add_argument('-d', '--datapoints', type=int, default=None, help='If set, specifies the number of training datapoints')
    args = parser.parse_args()  

    if args.load:
        tokenizer, model = utils.load_model(args.load)
        user_interface(model, tokenizer)
    else:
        train(args.datapoints)
