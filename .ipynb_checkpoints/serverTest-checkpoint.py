import zerorpc

class SageBrain(object):
    def hello(self, name):
        return "Hello, %s" % name

s = zerorpc.Server(SageBrain())
s.bind("tcp://0.0.0.0:4242")
s.run()