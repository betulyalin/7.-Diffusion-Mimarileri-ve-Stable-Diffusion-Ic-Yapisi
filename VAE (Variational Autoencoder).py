"""
Bu kod MNIST veri seti üzerinde basit bir Variational Autoencoder (VAE) modeli eğitir.
Model, 28x28 boyutundaki el yazısı rakamları düşük boyutlu latent uzaya sıkıştırır ve
buradan tekrar orijinal görüntüyü yeniden üretmeye çalışır.
Eğitim tamamlandıktan sonra Gradio arayüzü ile kullanıcı, veri setinden istediği bir örneği
seçip orijinal ve VAE tarafından yeniden oluşturulan görüntüyü karşılaştırabilir.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import gradio as gr

# --- VAE Model ---
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(28*28, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 28*28)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 28*28))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# --- Loss function ---
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 28*28), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(latent_dim=20).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- Dataset ---
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# --- Training loop ---
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item() / len(data):.4f}")
    print(f"Epoch {epoch} average loss: {train_loss / len(train_loader.dataset):.4f}")

# --- Eğit ---
for epoch in range(1, 6):  # 5 epoch eğit, istersen artırabilirsin
    train(epoch)

# --- Gradio demo fonksiyonu ---
def vae_demo(idx):
    model.eval()
    with torch.no_grad():
        img, _ = train_dataset[idx]
        img = img.to(device).unsqueeze(0)
        recon, _, _ = model(img)
        recon_img = recon.view(28, 28).cpu()
        orig_img = img.view(28, 28).cpu()
        return orig_img, recon_img

# --- Arayüz ---
def vae_demo(idx):
    model.eval()
    with torch.no_grad():
        img, _ = train_dataset[idx]
        img = img.to(device).unsqueeze(0)
        recon, _, _ = model(img)
        recon_img = recon.view(28, 28).cpu()
        orig_img = img.view(28, 28).cpu()

        # 224x224 olarak büyüt
        orig_img_pil = transforms.ToPILImage()(orig_img).resize((224, 224))
        recon_img_pil = transforms.ToPILImage()(recon_img).resize((224, 224))

        return orig_img_pil, recon_img_pil

demo = gr.Interface(
    fn=vae_demo,
    inputs=gr.Slider(0, len(train_dataset)-1, step=1, label="MNIST Veri Setinden Örnek İndeksi"),
    outputs=[
        gr.Image(label="Orijinal"),
        gr.Image(label="VAE Rekonstrüksiyonu")
    ],
    title="MNIST VAE Demo",
    description="MNIST veri setinden bir örnek seç, VAE modelinin yeniden oluşturmasını gör."
)


demo.launch()
