import os
import math
import torch
import random
import numpy as np
import argparse
import json
import cohere
from openai import OpenAI

from src.models.lusifer import WrappedLusifer
from transformers import HfArgumentParser
from src.args import DataArguments, ModelArguments, TrainingArguments

from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from collections import Counter

from utils import NusaXDataset, NusaTranslationDataset, TatoebaDataset, BUCCDataset, LinceMTDataset, PhincDataset, LinceSADataset, MassiveIntentDataset, Sib200Dataset, NollySentiDataset, MTOPIntentDataset, FIREDataset

OPENAI_TOKEN = ""
COHERE_TOKEN = ""

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_openai_embedding(model, texts, checkpoint="text-embedding-3-large"):
    data = model.embeddings.create(input = texts, model=checkpoint).data
    embeddings = []
    for obj in data:
        embeddings.append(obj.embedding)
    return embeddings

def get_cohere_embedding(model, texts, model_checkpoint):
    response = model.embed(texts=texts, model=model_checkpoint, input_type="search_query")
    return response.embeddings

def evaluate_classification(train_embeddings, test_embeddings, train_labels, k):
    hyps = []
    for test_id in tqdm(range(len(test_embeddings))):
        dists = []
        batch_size = 128
        if len(train_embeddings) < batch_size:
            batch_size = len(test_embeddings) // 2
        
        num_of_batches = math.ceil(len(train_embeddings) / batch_size)

        for i in range(num_of_batches):
            train_embedding = torch.FloatTensor(train_embeddings[i*batch_size:(i+1)*batch_size]).unsqueeze(1).cuda()
            
            test_embedding = torch.FloatTensor(test_embeddings[test_id]).unsqueeze(0)
            test_embedding = test_embedding.expand(len(train_embedding), -1).unsqueeze(1).cuda()

            # print(train_embedding.size(), test_embedding.size())
            
            dist = torch.cdist(test_embedding, train_embedding , p=2, compute_mode='use_mm_for_euclid_dist_if_necessary').squeeze().tolist()

            if isinstance(dist, float):
                dist = [dist]

            for j in range(len(dist)):
                dists.append([dist[j], train_labels[i*batch_size + j]])

        sorted_dists = sorted(dists,key=lambda l:l[0], reverse=False)[:k]
        all_indices = [obj[1] for obj in sorted_dists]
        c = Counter(all_indices)
        majority = c.most_common()[0][0]
        hyps.append(majority)
    return hyps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        "--model_checkpoint",
        default=None,
        type=str,
        help="Path to pre-trained model")
    parser.add_argument("--cross", action="store_true")
    parser.add_argument("--src_lang", type=str, default="x", help="source language")
    parser.add_argument("--dataset", type=str, default="mtop", help="snips or mtop or multi-nlu")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--prompt", type=str, default="", help="prompt")
    args = parser.parse_args()

    # Make sure cuda is deterministic
    torch.backends.cudnn.deterministic = True

    print("###########################")
    print("src_lang:", args.src_lang)
    print("dataset:", args.dataset)
    print("model_name_or_path:", args.model_name_or_path)
    print("seed:", args.seed)
    print("cuda:", args.cuda)
    print("cross:", args.cross)
    print("verbose:", args.verbose)
    print("fp16:", args.fp16)
    print("prompt:", args.prompt)
    print("###########################")

    set_seed(args.seed)

    if args.cross:
        output_dir = f"outputs/save_classification_cross_{args.src_lang}"
    else:
        output_dir = "outputs/save_classification"

    if args.dataset == "nusax":
        prompt = 'Classify the sentiment expressed in the given text.'
        dataset = NusaXDataset(prompt=args.prompt, task="classification")
    if args.dataset == "lince_sa":
        prompt = 'Classify the sentiment expressed in the given text.'
        dataset = LinceSADataset(prompt=args.prompt)
    if args.dataset == "massive_intent":
        prompt = 'Given a user utterance as query, find the user intents.'
        dataset = MassiveIntentDataset(prompt=args.prompt)
    if args.dataset == "sib200":
        prompt = 'Identify the category of the following passages.'
        dataset = Sib200Dataset(prompt=args.prompt)
    if args.dataset == "nollysenti":
        prompt = 'Classify the sentiment expressed in the given text.'
        dataset = NollySentiDataset(prompt=args.prompt, task="classification")
    if args.dataset == "mtop_intent":
        prompt = 'Classify the intent of the given utterance in task-oriented conversation.'
        dataset = MTOPIntentDataset(prompt=args.prompt)
    if args.dataset == "fire":
        prompt = 'Classify the sentiment expressed in the given text.'
        dataset = FIREDataset(prompt=args.prompt)
    
    is_complete = True
    if args.model_checkpoint is not None:
        model_name = "Lusifer"
    else:
        model_name = args.model_name_or_path
    for lang in dataset.LANGS:
        if args.cross and lang == args.src_lang:
            print("skip src language eval", lang)
            continue
        
        # Check if embeddings are already computed
        if os.path.exists(f"{output_dir}/{args.dataset}/{model_name}/seed_{args.seed}/eval_{lang}_1.json"):
            print("skip", lang)
            continue
        else:
            is_complete = False
            break
    if is_complete:
        print("All languages are already evaluated")
        exit(0)

    is_lusifer = False
    if args.model_checkpoint is not None:
        is_lusifer = True
        args.prompt = ''
        config_file = args.model_name_or_path
        args.model_name_or_path = "Lusifer"
        model_checkpoint = args.model_checkpoint
        hf_parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
        print(f"Loading yaml config {config_file}")
        data_args, model_args, training_args = hf_parser.parse_yaml_file(yaml_file=config_file)
        model = WrappedLusifer(
            universal_learner_name_or_path=model_args.universal_learner_name_or_path,
            encoder_name_or_path=model_args.encoder_name_or_path,
            universal_learner_backbone_type=model_args.universal_learner_backbone_type,
            encoder_backbone_type=model_args.encoder_backbone_type,
            is_freeze_universal_learner=model_args.is_freeze_universal_learner,
            is_freeze_encoder=True,
            connection_type=model_args.connection_type,
            num_added_tokens=model_args.num_added_tokens,
            encoder_lora_name=model_args.encoder_lora_name,
            universal_learner_lora_name=model_args.universal_learner_lora_name,
            loar_r=model_args.loar_r,
            lora_alpha=model_args.lora_alpha,
            dropout=model_args.dropout,
            attn_implementation=model_args.attn_implementation,
            model_dtype= torch.bfloat16,
            model_revision=training_args.model_revision,
            model_checkpoint=model_checkpoint,
            num_gpus=8,
        )
        batch_size = 512
    elif "embed-multilingual" in args.model_name_or_path:
        model = cohere.Client(COHERE_TOKEN)
        batch_size = 64
    elif "text-embedding-3-large" in args.model_name_or_path:
        model = OpenAI(api_key=OPENAI_TOKEN)
        batch_size = 64
    else:
        model = SentenceTransformer(args.model_name_or_path).cuda()
        batch_size = 128

    for lang in dataset.LANGS:
        if args.cross and lang == args.src_lang:
            print("skip src language eval", lang)
            continue
        
        # Check if embeddings are already computed
        if os.path.exists(f"{output_dir}/{args.dataset}/{args.model_name_or_path}/seed_{args.seed}/eval_{lang}_1.json"):
            print("skip", lang)
            continue

        # get embeddings
        if args.cross:
            train_texts = dataset.train_data[args.src_lang]["source"]
            train_labels = dataset.train_data[args.src_lang]["target"]
        else:
            train_texts = dataset.train_data[lang]["source"]
            train_labels = dataset.train_data[lang]["target"]

        if len(train_texts) < batch_size:
            batch_size = len(train_texts) // 2
        num_of_batches = math.ceil(len(train_texts) / batch_size)

        if args.cross:
            print("> train:", args.src_lang, num_of_batches)
        else:
            print("> train:", lang, num_of_batches)
        train_embeddings = []
        for i in tqdm(range(num_of_batches)):
            train_batch_text = train_texts[i*batch_size:(i+1)*batch_size]
            train_batch_label = train_labels[i*batch_size:(i+1)*batch_size]

            if is_lusifer:
                train_batch_embeddings = model.encode_queries(train_batch_text, instruction=prompt)
            elif "embed-multilingual" in args.model_name_or_path:
                train_batch_embeddings = get_cohere_embedding(model, train_batch_text, args.model_name_or_path)
            elif "text-embedding-3-large" in args.model_name_or_path:
                train_batch_embeddings = get_openai_embedding(model, train_batch_text, args.model_name_or_path)
            else:
                train_batch_embeddings = model.encode(train_batch_text, normalize_embeddings=False)

            if len(train_embeddings) == 0:
                train_embeddings = train_batch_embeddings
            else:
                for emb in train_batch_embeddings:
                    train_embeddings = np.concatenate((train_embeddings, np.expand_dims(emb, axis=0)), axis=0)

        # test
        test_texts = dataset.test_data[lang]["source"]
        test_labels = dataset.test_data[lang]["target"]

        if len(test_texts) < batch_size:
            batch_size = len(test_texts) // 2
        num_of_batches = math.ceil(len(test_texts) / batch_size)
            
        print("> test:", lang, num_of_batches)
        test_embeddings = []
        for i in tqdm(range(num_of_batches)):
            test_batch_text = test_texts[i*batch_size:(i+1)*batch_size]
            test_batch_label = test_labels[i*batch_size:(i+1)*batch_size]

            if is_lusifer:
                test_batch_embeddings = model.encode_queries(test_batch_text, instruction=prompt)
            elif "embed-multilingual" in args.model_name_or_path:
                test_batch_embeddings = get_cohere_embedding(model, test_batch_text, args.model_name_or_path)
            elif "text-embedding-3-large" in args.model_name_or_path:
                test_batch_embeddings = get_openai_embedding(model, test_batch_text, args.model_name_or_path)
            else:
                test_batch_embeddings = model.encode(test_batch_text, normalize_embeddings=False)

            if len(test_embeddings) == 0:
                test_embeddings = test_batch_embeddings
            else:
                for emb in test_batch_embeddings:
                    test_embeddings = np.concatenate((test_embeddings, np.expand_dims(emb, axis=0)), axis=0)

        if not os.path.exists(f"{output_dir}/{args.dataset}/{args.model_name_or_path}/seed_{args.seed}/"):
            os.makedirs(f"{output_dir}/{args.dataset}/{args.model_name_or_path}/seed_{args.seed}/")

        for k in [1,5,10]:
            key = lang
            print(key, k, train_embeddings.shape, test_embeddings.shape)
            hyps = evaluate_classification(train_embeddings, test_embeddings, train_labels, k=k)
            # print(hyps)
            # print(test_labels)
            obj = {}
            obj[f'acc'] = accuracy_score(test_labels, hyps)
            obj[f'prec'] = precision_score(test_labels, hyps, average="macro")
            obj[f'rec'] = recall_score(test_labels, hyps, average="macro")
            obj[f'f1'] = f1_score(test_labels, hyps, average="macro")
            print(obj)

            file_path = output_dir + "/" + args.dataset + "/" + args.model_name_or_path + "/" + "/seed_" + str(args.seed) + "/eval_" + key + "_" + str(k) + ".json"
            print("writing results to file_path:", file_path)
            with open(file_path, "w") as outfile: 
                json.dump(obj, outfile, indent=4)