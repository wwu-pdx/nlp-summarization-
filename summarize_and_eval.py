from summarizer import Summarizer,TransformerSummarizer
from datasets import  load_dataset
from rouge import Rouge
import random
import torch
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration,pipeline

rouge = Rouge()
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-xsum')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-xsum')
#tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
#model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

summarizer_pipe = pipeline("summarization")
summarizer_tune= pipeline("summarization", model="facebook/bart-large-xsum", tokenizer="facebook/bart-large-xsum", framework="pt")
#summarizer_tune= pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn", framework="pt")


GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
bert_model = Summarizer()
datasets = load_dataset('xsum')
#datasets = load_dataset('cnn_dailymail','3.0.0')



num = len(datasets["test"])
ids = random.sample(range(0, num-1), 1000)

#rouge = datasets.load_metric("rouge")

references=[]
summaries_gpt2=[]
summaries_bert=[]
summaries_bart=[]
summaries_pipe=[]
summaries_tune=[]
for i in range(0,len(ids)):
    body= datasets["test"]["document"][ids[i]]
    ref= datasets["test"]["summary"][ids[i]]
    #body= datasets["test"]["article"][ids[i]]
    #ref= datasets["test"]["highlights"][ids[i]]
    references.append(ref)
    
    # Pileine bart -fine tuned on summerization tasks, this gives the best rouge scores
    summary_tune=summarizer_tune(body, min_length=60, max_length=150,truncation=True)[0]['summary_text']
    summaries_tune.append(summary_tune)

    #bart with pre=trained tokens
    #This will fail on GPU becuase input and output are not on the same device. Have not figure out how to move strs around. 
    #Ideally, we want to use the whole datasets instead of subsets, so this model can be put on a seperate run. 
    article_input_ids = tokenizer.batch_encode_plus([body], return_tensors='pt', max_length=1024,truncation=True)['input_ids'].to(torch_device)
    summary_ids = model.generate(article_input_ids,num_beams=4, length_penalty=2.0, min_length=60,no_repeat_ngram_size=3)
    summary_bart = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
    summaries_bart.append(summary_bart)
    
    #pipeline bart
    summary_pipe=summarizer_pipe(body, min_length=60,max_length=150,truncation=True)[0]['summary_text']
    summaries_pipe.append(summary_pipe)

    #gpt
    summary_gpt2 = ''.join(GPT2_model(body, min_length=60,max_length=150))
    summaries_gpt2.append(summary_gpt2)
    #bert
    summary_bert = ''.join(bert_model(body, min_length=60,max_length=150))
    summaries_bert.append(summary_bert)


score_tune=rouge.get_scores(summaries_tune, references,avg=True)
print("score_tune")
print(score_tune)

score_bart=rouge.get_scores(summaries_bart, references,avg=True)
print("score_bart")
print(score_bart)

score_pipe=rouge.get_scores(summaries_pipe, references,avg=True)
print("score_pipe")
print(score_pipe)

score_gpt2=rouge.get_scores(summaries_gpt2, references,avg=True)
print("score_gpt2")
print(score_gpt2)

score_bert=rouge.get_scores(summaries_bert, references,avg=True)
print("score_bert")
print(score_bert)

