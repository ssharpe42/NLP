

from spacy.en import English
import re
from collections import Counter
from numpy import log2, log10, log, exp, sqrt, array, ones, linalg
import pandas as pd
from pandas import Series, DataFrame
from functools import partial
import multiprocessing as multi
from scipy import stats
import matplotlib.pyplot as plt
import math




class Model:

	def __init__(self, corpus):

		self.tokenize = English()
		self.__preprocess(corpus)
		self.__set_ngram_stats(True)

	def __preprocess(self, corpus):

		self.sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', corpus.strip())
		self.token_sent = [self.tokenize(s) for s in self.sentences]
		self.word_sent = [['<s>','<s>']+ [w.orth_ for w in s] + ['</s>'] for s in self.token_sent]
		self.pos_sent = [['<s>','<s>']+[w.tag_ for w in s]+ ['</s>'] for s in self.token_sent]

	def get_sent_lists(self):
		return self.word_sent, self.pos_sent

	def get_prob_dist(self,lang):
		return self.lang['uni_p'],self.lang['bi_p'],self.lang['tri_p'] if lang else self.pos['uni_p'],self.pos['bi_p'],self.pos['tri_p']

	def ngrams(self,text_list,n):

		return list(zip(*[text_list[i:] for i in range(n)]))

	def __set_ngram_stats(self, lang):

		sent_list = self.word_sent if lang else self.pos_sent
		unigrams = [uni for s in sent_list for uni in self.ngrams(s[1:],1)]
		uni_count = Counter(unigrams)
		bigrams = [bi for s in sent_list for bi in self.ngrams(s,2)]
		bi_count = Counter(bigrams)
		trigrams= [tri for s in sent_list for tri in self.ngrams(s,3)]
		tri_count = Counter(trigrams)
		counts = {'1':uni_count, '2':bi_count, '3':tri_count}

		if lang:
			print('Calculating language model...')
			self.lang = {'unigrams':unigrams, 'bigrams':bigrams, 'trigrams':trigrams,
						'counts':{1:uni_count, 2:bi_count, 3:tri_count}}
			self.lang['gt_counts'] = dict(zip([1,2,3],self.__GoodTuring(lang)))
			self.lang['uni_p'] = {unigram:log2(self.lang['gt_counts'][1][unigram]/float(len(self.lang['unigrams']))) for unigram in self.lang['gt_counts'][1] }
			self.lang['bi_p'] = {bigram:log2(self.lang['gt_counts'][2][bigram]/float(self.lang['gt_counts'][1][bigram[:-1]])) for bigram in self.lang['gt_counts'][2] }
			self.lang['tri_p'] = {trigram:log2(self.lang['gt_counts'][3][trigram]/float(self.lang['gt_counts'][2][trigram[:-1]])) for trigram in self.lang['gt_counts'][3] }

		else:
			print('Calculating part of speech model...')
			self.pos = {'unigrams':unigrams, 'bigrams':bigrams, 'trigrams':trigrams,
						'counts':{1:uni_count, 2:bi_count, 3:tri_count}}
			self.pos['gt_counts'] = dict(zip([1,2,3],self.__GoodTuring(lang)))
			self.pos['uni_p'] = {unigram:log2(self.pos['gt_counts'][1][unigram]/float(len(self.pos['unigrams'])))  for unigram in self.pos['gt_counts'][1] }
			self.pos['bi_p'] = {bigram:log2(self.pos['gt_counts'][2][bigram]/float(self.pos['gt_counts'][1][bigram[:-1]])) for bigram in self.pos['gt_counts'][2] }
			self.pos['tri_p'] = {trigram:log2(self.pos['gt_counts'][3][trigram]/float(self.pos['gt_counts'][2][trigram[:-1]])) for trigram in self.pos['gt_counts'][3] }



	def __GoodTuring(self, lang):

		def nbins(data):
			uniN = Counter(data['counts'][1].values())
			biN = Counter(data['counts'][2].values())
			triN = Counter(data['counts'][3].values())
			N = {1:uniN,2:biN,3:triN}
			return N

		def get_Z(frequencies,Nbins):

			sorted_freq = sorted(frequencies)
			Z = {}
			for i,j in enumerate(sorted_freq):
				if i == 0:
					q = 0
				else:
					q = sorted_freq[i-1]
				if i == len(sorted_freq)-1:
					t = 2*j - q
				else:
					t = sorted_freq[i+1]
				Z[j] = 2*Nbins[j]/float(t-q)

			return Z

		def log_linear_reg(x, y):
			X = array([ log(x), ones(len(x))])
			b,a = linalg.lstsq(X.T,log(y))[0]
			return b, a

		def smooth(a, b, Nbins, data, order, conf):

			N = {}
			insignif = False

			for r in Nbins[order]:

				Nr = Nbins[order][r]
				#log linear smoothied estimate
				smoothed_est = (r+1)*exp(b*log(r+1)+a)/exp(b*log(r)+a)
				if (r+1) not in Nbins[order]:
					insignif = True

				if not insignif:
					Nr1 = Nbins[order][r+1]
					#Turing heuristic estimate
					turing_est = (r+1)*Nr1/Nr
					#Standard deviation of estimate
					sd = ((r+1)**2)*(Nr1/Nr**2)*(1 + (Nr1/Nr))

					#test our smoothed estimate is outside confidence interval we
					if abs(smoothed_est - turing_est) <= sd*conf:
						insignif = True
					else:
						N[r] = turing_est

				if insignif:
					N[r] = smoothed_est

			p0 = Nbins[order][1]/float(sum(Nbins[order].values())) if Nbins[order][1]>0 else 2**-1000

			smoothed_counts = {}
			original_count = sum(data['counts'][order].values())
			Total = sum([Nbins[order][r]*val for r,val in N.items()])
			for ngram, count in data['counts'][order].items():
				smoothed_counts[ngram] = (1-p0)*N[count]*original_count/Total
			smoothed_counts[('<unk>',)*order] = p0*original_count

			return smoothed_counts

		data = self.lang if lang else self.pos
		Nbins = nbins(data)
		Z1 = get_Z(list(Nbins[1].keys()),Nbins[1])
		Z2 = get_Z(list(Nbins[2].keys()),Nbins[2])
		Z3 = get_Z(list(Nbins[3].keys()),Nbins[3])

		b1, a1 = log_linear_reg(list(Z1.keys()), list(Z1.values()))
		b2, a2 = log_linear_reg(list(Z2.keys()), list(Z2.values()))
		b3, a3 = log_linear_reg(list(Z3.keys()), list(Z3.values()))

		return smooth(a1,b1,Nbins,data,1,1.96), smooth(a2,b2,Nbins,data,2,1.96), smooth(a3,b3,Nbins,data,3,1.96)

######### TO DO ###################
	def apply_model(self, corpus):

		#this will eventually come in preprocessed from the other book, but we do it here for now
		sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', corpus.strip())
		token_sent = [self.tokenize(s) for s in sentences]
		word_sent = [['<s>','<s>']+ [w.orth_ for w in s] + ['</s>'] for s in token_sent]
		pos_sent = [['<s>','<s>']+[w.tag_ for w in s]+ ['</s>'] for s in token_sent]

		return sum(self.linearscore(word_sent,True))/len([w for s in word_sent for w in s[2:]]), sum(self.linearscore(pos_sent, False))/len([w for s in pos_sent for w in s[2:]])


	def linearscore(self, sent_token_list, lang):

		bigram_sent_list = [self.ngrams(sent,2) for sent in sent_token_list]
		trigram_sent_list = [self.ngrams(sent,2) for sent in sent_token_list]

		if lang:
			uni_scores = [[(1.0/3)*pow(2,self.lang['uni_p'][(word,)]) if (word,) in self.lang['uni_p'] else self.lang['uni_p'][('<unk>',)]/3.0 for word in s[2:]] for s in sent_token_list ]
			bi_scores = [[(1.0/3)*pow(2,self.lang['bi_p'][bigram]) if bigram in self.lang['bi_p'] else self.lang['bi_p'][('<unk>','<unk>')] for bigram in s[1:]] for s in bigram_sent_list ]
			tri_scores = [[(1.0/3)*pow(2,self.lang['tri_p'][trigram]) if trigram in self.lang['tri_p'] else self.lang['tri_p'][('<unk>','<unk>','<unk>')] for trigram in s] for s in trigram_sent_list ]

		else:
			uni_scores = [[(1.0/3)*pow(2,self.pos['uni_p'][(word,)]) if (word,) in self.pos['uni_p'] else self.pos['uni_p'][('<unk>',)]/3.0 for word in s[2:]] for s in sent_token_list ]
			bi_scores = [[(1.0/3)*pow(2,self.pos['bi_p'][bigram]) if bigram in self.pos['bi_p'] else self.pos['bi_p'][('<unk>','<unk>')] for bigram in s[1:]] for s in bigram_sent_list ]
			tri_scores = [[(1.0/3)*pow(2,self.pos['tri_p'][trigram]) if trigram in self.pos['tri_p'] else self.pos['tri_p'][('<unk>','<unk>','<unk>')] for trigram in s] for s in trigram_sent_list ]


		combined_scores = [list(zip(uni_scores[i],bi_scores[i],tri_scores[i])) for i in range(0,len(uni_scores))]
		scores = [sum([math.log(sum(score_tuple),2) for score_tuple in s]) for s in combined_scores]

		return scores






def main():

	#Brown = open('/Users/Sam/Desktop/School/NLP/Homework1/Brown_train.txt','r').read()
	Ulysess = open('/Users/Sam/Desktop/Projects/BookNLP/4300.txt','r').read()
	Ulysess=re.sub('\n',' ',Ulysess.replace('--',''))
	Ulysess = re.sub('\t',' ',Ulysess)
	Ulysess = ' '.join(Ulysess.split())[550:]

	HeartofDarkness = open('/Users/Sam/Desktop/Projects/BookNLP/HeartofDarkness.txt','r').read()
	HeartofDarkness = HeartofDarkness[660:]
	HeartofDarkness=re.sub('\n',' ',HeartofDarkness)
	HeartofDarkness = re.sub('\t',' ',HeartofDarkness)

	ThisSideofParadise = open('/Users/Sam/Desktop/Projects/BookNLP/ThisSideofParadise.txt','r').read()
	ThisSideofParadise = ThisSideofParadise[675:]
	ThisSideofParadise=re.sub('\n',' ',ThisSideofParadise)
	ThisSideofParadise = re.sub('\t',' ',ThisSideofParadise)


	PrideandPrejudice = open('/Users/Sam/Desktop/Projects/BookNLP/PrideandPrejudice.txt','r').read()
	PrideandPrejudice = PrideandPrejudice[651:]
	PrideandPrejudice=re.sub('\n',' ',PrideandPrejudice)
	PrideandPrejudice = re.sub('\t',' ',PrideandPrejudice)


	M = Model(Ulysess)
	HOD = M.apply_model(HeartofDarkness)
	TSOP =M.apply_model(ThisSideofParadise)
	PAP= M.apply_model(PrideandPrejudice)

	HODw = .5*2**HOD[0] + .5*2**HOD[1]
	TSOPw = .5*2**TSOP[0] + .5*2**TSOP[1]
	PAPw = .5*2**PAP[0] + .5*2**PAP[1]




if __name__ == "__main__":
    main()
