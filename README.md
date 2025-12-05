<p align="center">
  <img src="https://img.shields.io/badge/EP--GAN-Edge%20Preserving%20Underwater%20Image%20Restoration-blue?style=for-the-badge" />
</p>

<p align="center">
  <strong>EP-GAN: Edge-Preserving Generative Adversarial Network for Underwater Image Restoration</strong><br>
  <em>A Deep Learning Framework for High-Fidelity Underwater Image Enhancement</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c?logo=pytorch" />
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg"/>
  <img src="https://img.shields.io/badge/Status-Research%20Code-orange"/>
</p>

---

# **EP-GAN: Edge-Preserving Generative Adversarial Network for Underwater Image Restoration**

This repository provides the official implementation of **EP-GAN**, a deep learning framework designed to restore underwater images while preserving structural edges, fine textures, and global perceptual quality.
The model synergizes **edge-aware priors**, **multi-scale GAN design**, and **feature-consistent learning**, making it suitable for real-world underwater image enhancement.

The full source code is archived under a public DOI for transparency and reproducibility.

---

# 🧭 **Overview**

Underwater images often suffer from:

* Color distortion
* Low contrast
* Scattering & absorption effects
* Blurred or missing edges

EP-GAN introduces:

✔ Edge-guided generator (RGB + Canny)
✔ Multi-scale discriminator
✔ Residual + deformable blocks
✔ Attention-driven feature fusion
✔ Perceptual + content + edge + GAN losses

---

# 📚 **Repository Structure**

```
EPGAN/
│
├── model/                # Generator, Discriminator & network blocks
├── database/             # EUVP & UIEB dataset loaders
├── functions/            # Edge detection, metrics, utilities
├── objectives/           # All loss functions
│
├── train.py              # Training script (80/20 split)
├── test.py               # Evaluation + visualization
│
├── notebooks/
│   ├── EUVP_Experiment.ipynb
│   └── UIEB_Experiment.ipynb
│
├── requirements.txt
├── LICENSE
└── README.md
```
---
# 📌**Architecture**
<p align="center">
<img width="400" height="600" alt="image" src="https://github.com/user-attachments/assets/937f9937-afeb-403f-8803-249cff8345b7" />
<img  width="400" height="700" alt="image"  alt="image" src="https://github.com/user-attachments/assets/9556395d-35a2-4d9b-a6b8-ad8171a20214" />
</p>
<p align="center">
<img width="300" height="500"  alt="image" src="https://github.com/user-attachments/assets/032cb645-83b4-4592-a6a0-8bf438c66c4f" />
<img  width="300" height="500"  alt="image" src="https://github.com/user-attachments/assets/68af105b-43f9-4b4c-a931-867e33862fdf" />
<img  width="300" height="500"  alt="image" src="https://github.com/user-attachments/assets/ef1bb7b9-f6ce-48c5-b037-ccd949fa869f" />
  </p>
  
---

# 📌**Sample**
<p align="center">
<img width="3376" height="2312" alt="image" src="https://github.com/user-attachments/assets/69ab64e9-89cf-401e-b0ef-f0efae6f3241" />
</p>

---

# ⚡ **Quick Start**

### 1. Clone the Repo

```bash
git clone https://github.com/YOUR_USERNAME/EPGAN.git
cd EPGAN
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Datasets

* EUVP (paired only): [https://irvlab.cs.umn.edu/resources/euvp-dataset](https://irvlab.cs.umn.edu/resources/euvp-dataset)
* UIEB benchmark: [https://li-chongyi.github.io/proj_benchmark.html](https://li-chongyi.github.io/proj_benchmark.html)

### 4. Train on EUVP

```bash
python train.py --data_root datasets/EUVP/Paired
```

### 5. Test the Model

```bash
python test.py --data_root datasets/EUVP/Paired --checkpoint generator_best.pth
```

---

# 📂 **Dataset Folders**

### EUVP

```
datasets/EUVP/Paired/
   underwater_dark/trainA & trainB
   underwater_imagenet/trainA & trainB
   underwater_scenes/trainA & trainB
```

### UIEB

```
datasets/UIEB/raw/
datasets/UIEB/reference/
```

### Notes

* Only paired data is used
* 80/20 internal split applied
* EUVP & UIEB trained separately

---

# 🧪 **Evaluation Metrics**

EP-GAN computes:

* **PSNR**
* **SSIM**
* **Edge Preservation Metrics**
* **Visual comparisons** (Distorted → Edges → Restored → GT)

Test results saved as:

```
results/test_results.png
```

---

# 🔬 **Jupyter Notebooks Included**

* **EUVP_Experiment.ipynb** — full training & experiments
* **UIEB_Experiment.ipynb** — UIEB testing & visualization
* Exported HTML notebooks available under:

---

# 🔒 **Dataset Licensing Disclaimer**

This project **does NOT** distribute any dataset files.
All datasets belong to their original authors and are used **strictly for academic, non-commercial research**.

---

# 📖 **Citation**

If you use this code or methodology, please cite:

**“EP-GAN: An Edge Preserving Generative Adversarial Network for Underwater Image Restoration”**
*The Visual Computer (Springer)*, 2025.

---

# 📝 **BibTeX**

```bibtex
@article{EPGAN2025,
  title={EP-GAN: An Edge Preserving Generative Adversarial Network for Underwater Image Restoration},
  journal={The Visual Computer},
  year={2025},
  authors={Plawang Shishu, Sruthi Nair, Mayur Parate, Tausif Diwan, Parul Sahare}
}
```

---

# ⚖️ **License**

MIT License — free for academic research and experimentation.

---

