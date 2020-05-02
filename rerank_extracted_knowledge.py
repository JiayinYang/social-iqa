# Scikit Learn
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import math
import re
from collections import Counter
import numpy as np
import pickle
from tqdm import tqdm
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity

WORD = re.compile(r"\w+")
def redundant_score(candidate, score, sentence_vector, sentence):
	#s1 = tfidf_vectorizer.fit_transform(sentence[0].strip().split(" "))
	#s2 = tfidf_vectorizer.transform(sentence[candidate].strip().split(" "))
	s1 = text_to_vector(sentence[0])
	s2 = text_to_vector(sentence[candidate])
	red = get_cosine(s1, s2)
	
	#red = cosine_similarity(s1, s2)
	#red = 1 - spatial.distance.cosine(s1, s2)
	for i in range(len(sentence)):
		if i != candidate:
			#s1 = tfidf_vectorizer.fit_transform(sentence[i].strip().split(" "))
			#s2 = tfidf_vectorizer.transform(sentence[candidate].strip().split(" "))
			#cos = cosine_similarity(s1, s2)
			s1 = text_to_vector(sentence[i])
			s2 = text_to_vector(sentence[candidate])
			cos = get_cosine(s1, s2)
			#print(red)
			#print(cos)
			red = max(red, cos)
	return red

def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def rerank(score, sentence_vector,sentence):
	relevant_score = score
	for candidate in range(1, len(sentence)):
		red_score = redundant_score(candidate, score, sentence_vector, sentence)
		relevant_score[candidate] = (1.0 - red_score) * float(score[candidate])
	return relevant_score

def main(subset='trn'):
	f = open("socialIQa_v1.4_"+subset+"_knowledge50.txt", "r",encoding="utf-8")
	score = []
	sentence_vector = []
	sentence = []
	start = 0
	tfidf_vectorizer = TfidfVectorizer()
	j = 0
	answers = []
	scores = []
	for line in tqdm(f.readlines()):	
		if line.strip().startswith("##"):
			if start != 0:
				relevant_score = rerank(score, sentence_vector, sentence)
				indexlist = []
				scorelist = []
				if relevant_score == []:
					answers.append(indexlist)
					scores.append(scorelist)
				else:
					for i in range(10):
						index = relevant_score.index(max(relevant_score))
						scorelist.append(relevant_score[index])
						relevant_score[index] = -math.inf
						indexlist.append(sentence[index])
					answers.append(indexlist)
					scores.append(scorelist)
				score = []
				sentence = []
				sentence_vector = []
			start = 1
		else:
			newline = line.split("\t")
			score.append(float(newline[0]))
			sparse_matrix = tfidf_vectorizer.fit_transform(newline[1].strip().split(" "))
			sentence_vector.append(sparse_matrix)
			sentence.append(newline[1])
		j = j + 1


	relevant_score = rerank(score, sentence_vector, sentence)
	indexlist = []
	scorelist = []
	if relevant_score == []:
		answers.append(indexlist)
		scores.append(scorelist)
	else:
		for i in range(10):
			index = relevant_score.index(max(relevant_score))
			scorelist.append(relevant_score[index])
			relevant_score[index] = -math.inf
			indexlist.append(sentence[index])
		answers.append(indexlist)
		scores.append(scorelist)

	f = open("socialIQa_v1.4_"+subset+"_knowledge_reranked10.txt", "w",encoding='utf-8')
	for ans, score in zip(answers, scores):
		f.write(str(ans)+'\t'+str(score)+'\n')
	f.close()

	infile = open("socialIQa_v1.4_"+subset+"_knowledge50.pickle",'rb')
	new_dict = pickle.load(infile)
	infile.close()
	index = 0
	print(len(answers))
	print(len(new_dict))
	for i in tqdm(range(len(new_dict))):
		new_dict[i]["answerA"] = answers[index]
		index = index + 1
		new_dict[i]["answerB"] = answers[index]
		index = index + 1
		new_dict[i]["answerC"] = answers[index]
		index = index + 1

	output = open("socialIQa_v1.4_"+subset+"_knowledge_reranked10.pkl", 'wb')
	pickle.dump(new_dict, output)
	output.close()


if __name__ == "__main__":
	for subtype in ["trn","dev","tst"]:
		main(subtype)


# Create the Document Term Matrix
#count_vectorizer = CountVectorizer(stop_words='english')
#count_vectorizer = CountVectorizer()
#sparse_matrix = count_vectorizer.fit_transform(documents)

# OPTIONAL: Convert Sparse Matrix to Pandas Dataframe if you want to see the word frequencies.
#doc_term_matrix = sparse_matrix.todense()
#df = pd.DataFrame(doc_term_matrix, 
 #                 columns=count_vectorizer.get_feature_names(), 
  #                index=['doc_trump', 'doc_election', 'doc_putin'])
# k is word vectors
# rel is relevant score array




 
