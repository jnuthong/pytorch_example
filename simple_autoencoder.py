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

num_epochs = 100
batch_size = 64
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

class AutoEncoder(nn.Module):
	"""
	"""
	def __init__(self, latent_num=16):
		"""
		TODO: doconvolution
		"""
		super(AutoEncoder, self).__init__()
		
		self.fc1 = nn.Linear(IMG_SIZE, 256)
		self.fc1.weight.data.normal_(0.0, 0.05)

		self.fc2 = nn.Linear(256, 64)
		self.fc2.weight.data.normal_(0.0, 0.05)

		self.fc3 = nn.Linear(64, 256)
		self.fc3.weight.data.normal_(0.0, 0.05)

		self.fc4 = nn.Linear(256, IMG_SIZE)
		self.fc4.weight.data.normal_(0.0, 0.05)


	def forward(self, x):
		h1 = F.relu(self.fc1(x))  # IMG_SIZE -> 518
		h2 = F.relu(self.fc2(h1)) # 518 -> 256
		h3 = F.relu(self.fc3(h2)) # 256 -> 128
		h4 = F.relu(self.fc4(h3)) # 128 -> 256
		output = h4
		# output = F.sigmoid(h6)
		return output

# ref: http://kvfrans.com/variational-autoencoders-explained/
# 	1) encoder loss = mean square error from original image and decoder image
# 	2) decoder loss = KL divergence 
encoder_loss = nn.MSELoss(size_average=True)

def loss_function(output, x):
		"""
		"""
		mse = encoder_loss(output, x)
		return mse

# way to construct DNN network
# 	1) topology 
# 	2) loss function
#		3) optimizer
# 	4) forward
# 	5) free zero grad of variable
# 	6) backward

model = AutoEncoder()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
	train_loss = 0
	for batch_idx, data in enumerate(dataloader):
		img = data.view(data.size(0), -1)
		img = Variable(img.float())
		# free zero grad
		optimizer.zero_grad()
		output = model(img)
		# backward
		loss = loss_function(output, img)
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
		save = to_img(output.cpu().data)
		save_image(save, './vae_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './vae.pth')
