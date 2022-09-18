from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from summarizer import Summarizer, TransformerSummarizer

productQuestionsAnswers = '''
What is the price of this product.This will cost you around 20 $.What is the weight of the product ?.It is 10 pounds.
        '''


# bert_model = Summarizer()
# productSummary = ''.join(bert_model(productQuestionsAnswers, min_length=20))
# print(productSummary)



# GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
# full = ''.join(GPT2_model(productSummary, min_length=60))
# print(full)

n_gram_range = (1, 1)
stop_words = "english"

# Extract candidate words/phrases
count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([productQuestionsAnswers])
candidates = count.get_feature_names()

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
doc_embedding = model.encode([productQuestionsAnswers])
candidate_embeddings = model.encode(candidates)

top_n = 10
distances = cosine_similarity(doc_embedding, candidate_embeddings)
keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

print(keywords)
