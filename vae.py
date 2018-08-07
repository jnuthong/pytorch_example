from mnist import MNIST
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from torchvision.utils import save_image

# from torchvision.datasets import MNIST

mndata = MNIST('./../data/')
images, labels = mndata.load_training()

img_transform = transforms.Compose([
                transforms.ToTensor()
                ])

num_epochs = 20
batch_size = 128
learning_rate = 1e-3
IMG_SIZE = 784
IMG_WIDTH = 28
IMG_HEIGHT = 28

dataset = torch.tensor(images)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

class VAE(nn.Module):
	"""
	"""
	def __init__(self, latent_num=2):
		"""
		TODO: doconvolution
		"""
		super(VAE, self).__init__()
		
		self.fc1 = nn.Linear(IMG_SIZE, 256)
		self.fc21 = nn.Linear(256, 	16)
		self.fc22 = nn.Linear(256, 16)
		self.fc3 = nn.Linear(16, 256)
		self.fc4 = nn.Linear(256, 784)

	def reparametrize(self, mu, std):
		"""
		why we need reparameterize:
			- we want to learn p(z|x) distribution of latent variable given dataset
			- we want p(z|x) constraint on unit guassian
			- but we have fixed input (train-set), if we want model to be randomness, 
			- so we incoporate some noise ~ N(0, 1), and redirect this data to decoder layer
		"""
		eps = torch.FloatTensor(std.size()).normal_()
		eps = Variable(eps)
		return eps.mul(std).add_(mu)

	def encoder(self, x):
		h1 = F.relu(self.fc1(x))
		mu, std = self.fc21(h1), self.fc22(h1)
		return mu, std

	def decoder(self, x):
		h2 = F.relu(self.fc3(x))
		return self.fc4(h2) 
		
	def forward(self, x):
		mu, var = self.encoder(x)
		z = self.reparametrize(mu, var)
		return self.decoder(z), mu, var

# ref: http://kvfrans.com/variational-autoencoders-explained/
# 	1) encoder loss = mean square error from original image and decoder image
# 	2) decoder loss = KL divergence 
encoder_loss = nn.MSELoss(size_average=True)

def loss_function(output, x, mu, var):
		"""
		"""
		mse = encoder_loss(output, x)
		#	0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2) 
		latent_loss = mu.pow(2).add_(var.pow(2)).mul(-1.).add_(torch.log(var.pow(2))).add_(1).mul_(0.5)
		KLD = torch.sum(latent_loss)
		return mse - KLD

# way to construct DNN network
# 	1) topology 
# 	2) loss function
#		3) optimizer
# 	4) forward
# 	5) free zero grad of variable
# 	6) backward

model = VAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
	train_loss = 0
	for batch_idx, data in enumerate(dataloader):
		img = data.view(data.size(0), -1)
		img = Variable(img.float())
		# free zero grad
		optimizer.zero_grad()
		output, mu, var = model(img)
		# backward
		loss = loss_function(output, img, mu, var)
		loss.backward()
		train_loss += loss.data[0]
		optimizer.step()
		if batch_idx % 100 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(img),
                len(dataloader.dataset), 100. * batch_idx / len(dataloader),
                loss.data[0] / len(img)))
	print('====> Epoch: {} Average loss: {:.4f}'.format(
      	epoch, train_loss / len(dataloader.dataset)))
	if epoch % 10 == 0:
		save = to_img(img.cpu().data)
		save_image(save, './var_encoder_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './vae.pth')
