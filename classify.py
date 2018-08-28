import os,sys
import numpy as np
import re
from collections import defaultdict
from functions import get_filename
from functions import remove_stopwords

train_classes = []
train_classes_doc_count = []
train_documents_count = 0
train_cat_docslist = []


location = raw_input("Please enter the location of training root folder")

if ('train' in location):
	training_folder = 'train'
	train_classes ,train_classes_doc_count,train_cat_docslist = get_filename(training_folder)

location = raw_input("Please enter the location of test root folder")

if ('test' in location):
	test_folder = 'test'
	test_classes ,test_classes_doc_count,test_cat_docslist = get_filename(test_folder)


train_class_docs = []
train_total_docs = []
train_class_wcount = []
vocab_list =[]

# Removing stopwords

vocab_list, training_docs, train_total_docs, train_class_wcount = remove_stopwords(train_cat_docslist, training_folder, train_classes)
test_vocab_list, test_docs, test_total_docs, test_class_wcount = remove_stopwords(test_cat_docslist, test_folder, test_classes)


one_vocab_dict = defaultdict(int)
two_vocab_dict = defaultdict(int)
three_vocab_dict = defaultdict(int)
four_vocab_dict = defaultdict(int)
five_vocab_dict = defaultdict(int)

for x in vocab_list:
	one_vocab_dict[x] = 0
	two_vocab_dict[x] = 0
	three_vocab_dict[x] = 0
	four_vocab_dict[x] = 0
	five_vocab_dict[x] = 0


for n in range(0,5):
	for docs in training_docs[n]:
		for w in docs:		
			if n == 0:
				one_vocab_dict[w] += 1
		
			if  n == 1:
				two_vocab_dict[w] += 1
	
			if n == 2:
				three_vocab_dict[w] += 1
			
			if n == 3:
				four_vocab_dict[w] += 1
			
			if n == 4:
				five_vocab_dict[w] += 1
	

# Calucalating prior

prior = np.zeros(5)
for i in range(5):
	prior[i] = float(train_total_docs[i])/sum(train_total_docs)

true_value = []
predicted_value = []



# Calucalating likelihood

for n in range(0,5):	
	for docs in test_docs[n]:
		true_value.append(n)
		likelihood = [0,0,0,0,0] 
		for word in docs:

			if one_vocab_dict[word] != 0:
				likelihood[0] +=  np.log(float(one_vocab_dict[word]) + 1/(train_class_wcount[n] + len(vocab_list)))
		
			if two_vocab_dict[word] != 0:
				likelihood[1] +=  np.log(float(two_vocab_dict[word]) + 1/(train_class_wcount[n] + len(vocab_list)))

			if three_vocab_dict[word] != 0:
				likelihood[2] +=  np.log(float(three_vocab_dict[word])+ 1/(train_class_wcount[n] + len(vocab_list)))

			if four_vocab_dict[word] != 0:
				likelihood[3] +=  np.log(float(four_vocab_dict[word])+ 1/(train_class_wcount[n] + len(vocab_list)))

			if five_vocab_dict[word] != 0:
				likelihood[4] +=  np.log(float(five_vocab_dict[word])+ 1/(train_class_wcount[n] + len(vocab_list)))
		

		# Allocating document to class
		posterior = [0, 0, 0, 0, 0]
		for j in range(5):
			posterior[j] = float(likelihood[j] + np.log(prior[j]) )
		
		print '\nTesting... True Value:', n, 'Predicted Value:', posterior.index(max(posterior))
		predicted_value.append(posterior.index(max(posterior)))


# Calucalating accuracy
count = 0
for i in range(len(predicted_value)):
	if predicted_value[i] == true_value[i]:
		count = count + 1


print '\n Accuracy =', float(count)/ len(predicted_value)



