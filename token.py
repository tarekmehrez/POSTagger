from collections import defaultdict


class Token(object):

	def __init__(self):
		self.feats = defaultdict(float)


	def set_feat(self,feat):
		self.feats[feat] = 1

	def get_feats(self):
		return self.feats