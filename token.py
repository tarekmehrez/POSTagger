class Token(object):

	def __init__(self,form):
		self.feats = {}
		self.feats["FORM"] = form

	def set_feat(self,feat,val):
		self.feats[feat] = val

	def get_feats(self):
		return self.feats