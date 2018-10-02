from tensorboard import SummaryWriter

class Training:
	def __init__(self, Net, logDr):
		self.model = Net
		self.logDr = logDr

	def train(self, trainloader, epoch, lr=0.1):
		learning_rate = lr
		criterion = torch.nn.MSELoss()
		log = SummaryWriter(self.logDr)
		for i in range(epoch):
			count = 0
		    for data, target in trainloader:
		        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
		        learning_rate *= 0.9999
		        y_pred = self.model(t)
		        y = torch.zeros(t.shape[0], 10)
		        arr = torch.arange(t.shape[0])
		        y[arr,target] = 1.0
		        y = y.to(device)
		        loss = criterion(y_pred, y)
		        logger.add_scalar('loss on epoch ' + str(epoch) + ' ', loss.item(), count)
		        count += 1
		        optimizer.zero_grad()
		        loss.backward()
		        optimizer.step()
