import os,sys
import stop_words
from stop_words import get_stop_words
import numpy as np
import re
from collections import defaultdict

def get_filename(folder):
	"""Return the filenames in the given directory.

	Args:
    	folder: contains name of folder

    Returns:
    	Return the filenames in the given directory.
	"""
	classes = []
	classes_doc_count = []
	documents_count = 0	
	class_docslist = []
	folderlist = []

	for folder_name, subfolder_list, file_list in os.walk(folder):

		if len(subfolder_list) > 0:
			classes = subfolder_list
		
		if len(subfolder_list) == 0:
			documents_count = len(file_list)
			classes_doc_count.append(documents_count)
	
		if len(subfolder_list) == 0:
			class_docslist.append(file_list)
			folderlist.append(folder_name)
	
	return classes, classes_doc_count, class_docslist


	
def remove_stopwords(class_docslist, folder, classes):
	"""Reads all the documents in the given directory and preprocess the text data and generate tokens of words

	Args:
		class_docslist: list of class's documents
    	folder: contains name of folder
    	classes: list of classes 

    Returns: Returns vocab_list, class_docs, total_docs, class_wcount
	"""
	class_wcount_dict = {}
	total_docs = []
	class_wcount = []
	vocab_list = []
	class_docs = []

	for index, dlist in enumerate(class_docslist):
		class_word_count = 0
		total_docs_class = 0
		words_docs = []
		for d in dlist:
			print '\n Reading...'
			total_docs_class = len(dlist) 
			f = open('{0}/{1}/{2}'.format(folder,classes[index],d),'r')
		
			word_list = f.read()
			word_list = re.sub('[^A-Za-z]+', ' ', word_list)
			word_list = re.sub(r'\b\w{1,2}\b', '', word_list)
			word_list = re.sub(r'\w*\d\w*', '', word_list).strip()
			word_list = re.findall(r"[\w']+", word_list)
			w = [i.lower() for i in word_list]
			filtered_words = [x for x in w if x not in get_stop_words('english')]

			class_word_count += len(filtered_words) 

			for word in filtered_words:
				if word not in vocab_list:
					vocab_list.append(word)
	
			words_docs.append(filtered_words)
		total_docs.append(total_docs_class)
		class_docs.append(words_docs)
		class_wcount_dict[classes[index]] = class_word_count
		class_wcount.append(class_word_count)

	return vocab_list, class_docs, total_docs, class_wcount
	
