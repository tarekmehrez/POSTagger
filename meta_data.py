class MetaData(object):

	def __init__(self, vocab_file, labels_file):

		self.suffixes = ["able", "al", "ed", "ing", "er", "est", "ion", "ive", "less", "ly", "ness", "ous", "." ]

		with open(vocab_file) as file:
			self.vocab = [line.strip().decode('utf-8') for line in file]

		with open(labels_file) as file:
			self.labels = [line.strip().decode('utf-8') for line in file]

		self.feat_len = len(self.vocab) + len(self.suffixes) + 2

	def get_meta_data(self):
		return (self.vocab, self.labels, self.suffixes)
