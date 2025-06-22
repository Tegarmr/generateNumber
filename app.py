# =============================================================================
# Problem 3: Handwritten Digit Generation - Streamlit Web App
#
# Framework: Streamlit
# Model: Conditional Variational Autoencoder (CVAE)
#
# Instruksi:
# 1. Pastikan Anda sudah mengunduh 'cvae_mnist.pth' dari Google Colab.
# 2. Buat folder baru di komputer Anda.
# 3. Simpan file ini sebagai 'app.py' di dalam folder tersebut.
# 4. Letakkan file 'cvae_mnist.pth' di folder yang sama.
# 5. Buat file 'requirements.txt' di folder yang sama (isinya ada di bawah).
# 6. Buka terminal atau command prompt, navigasi ke folder tersebut, lalu jalankan:
#    pip install -r requirements.txt
#    streamlit run app.py
# 7. Aplikasi akan terbuka di browser Anda.
#
# Untuk deployment (agar bisa diakses publik):
# - Upload folder ini (app.py, cvae_mnist.pth, requirements.txt) ke GitHub.
# - Hubungkan repository GitHub Anda ke Streamlit Community Cloud (share.streamlit.io).
# =============================================================================

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# --- Konfigurasi dan Definisi Model ---

# Parameter harus sama dengan saat training
latent_dim = 16
num_classes = 10
image_size = 28 * 28
model_path = 'cvae_mnist.pth'

# Set device (CPU sudah cukup untuk inferensi)
device = torch.device("cpu")

# Arsitektur CVAE (harus sama persis dengan yang di training script)
class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super(CVAE, self).__init__()
        combined_input_dim = input_dim + num_classes
        self.encoder = nn.Sequential(
            nn.Linear(combined_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_log_var = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y_one_hot):
        x_combined = torch.cat([x, y_one_hot], dim=1)
        h = self.encoder(x_combined)
        mu, log_var = self.fc_mu(h), self.fc_log_var(h)
        z = self.reparameterize(mu, log_var)
        z_combined = torch.cat([z, y_one_hot], dim=1)
        return self.decoder(z_combined), mu, log_var

# --- Fungsi untuk Memuat Model dan Generate Gambar ---

# Cache model agar tidak di-load ulang setiap kali ada interaksi
@st.cache_resource
def load_model():
    model = CVAE(image_size, latent_dim, num_classes).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Set model ke mode evaluasi
        return model
    except FileNotFoundError:
        st.error(f"Error: File model '{model_path}' tidak ditemukan.")
        st.error("Pastikan file tersebut ada di folder yang sama dengan app.py.")
        return None

def generate_images(model, selected_digit, num_images=5):
    """Menghasilkan gambar berdasarkan digit yang dipilih."""
    if model is None:
        return []
        
    with torch.no_grad():
        # Buat noise acak dari distribusi normal
        z = torch.randn(num_images, latent_dim).to(device)
        
        # Buat label one-hot untuk digit yang dipilih
        c = torch.zeros(num_images, num_classes).to(device)
        c[:, selected_digit] = 1
        
        # Gabungkan noise dan label
        z_c = torch.cat([z, c], dim=1)
        
        # Generate gambar menggunakan decoder
        generated_samples = model.decoder(z_c).cpu()
        
        # Ubah tensor menjadi format gambar yang bisa ditampilkan
        images = []
        for sample in generated_samples:
            img_array = sample.view(28, 28).numpy()
            img_array = (img_array * 255).astype(np.uint8)
            images.append(Image.fromarray(img_array, 'L'))
            
        return images

# --- Antarmuka Pengguna (UI) Streamlit ---

st.set_page_config(layout="wide")
st.title("🖌️ Handwritten Digit Image Generator")
st.markdown("Aplikasi ini menghasilkan gambar tulisan tangan angka (seperti dataset MNIST) menggunakan model *Conditional VAE* yang dilatih dengan PyTorch.")

# Muat model
model = load_model()

if model:
    st.sidebar.header("Pengaturan")
    # Pilihan untuk user memilih digit
    digit_to_generate = st.sidebar.selectbox(
        "Pilih angka yang ingin dibuat (0-9):",
        options=list(range(10))
    )

    # Tombol untuk generate
    if st.sidebar.button("✨ Buat Gambar!", use_container_width=True):
        st.subheader(f"Gambar yang dihasilkan untuk angka: **{digit_to_generate}**")

        # Buat placeholder kolom
        cols = st.columns(5)
        
        # Tampilkan spinner saat proses generate
        with st.spinner('Sedang membuat gambar...'):
            generated_images = generate_images(model, digit_to_generate, num_images=5)
        
        # Tampilkan gambar di setiap kolom
        if generated_images:
            for i, (col, img) in enumerate(zip(cols, generated_images)):
                with col:
                    st.image(img, caption=f'Sample {i+1}', width=150, use_column_width='auto')
        else:
            st.warning("Gagal membuat gambar.")
else:
    st.warning("Model tidak dapat dimuat. Aplikasi tidak bisa berjalan.")
