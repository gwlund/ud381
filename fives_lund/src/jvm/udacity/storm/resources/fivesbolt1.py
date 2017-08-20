import storm

class fivesbolt1(storm.BasicBolt):

	def initialize(self, conf, context):
		self._conf = conf
		self._context = context

	def process(self, tup):
		word = tup.values[0]
		word+=str("???")
		storm.emit([word])

fivesbolt1().run()
