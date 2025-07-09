# 🧠 CIFAR-10 Image Classification with CNN (PyTorch)

A simple yet effective Convolutional Neural Network (CNN) built using PyTorch to classify images from the CIFAR-10 dataset. This project demonstrates how to train, evaluate, and save a deep learning model on real-world image data.

---

## 📚 Dataset: CIFAR-10

- 60,000 32x32 color images
- 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- 50,000 training images and 10,000 test images

> Downloaded automatically from `torchvision.datasets.CIFAR10`.

---

## 🛠️ Project Structure

```
📁 ImageClassification-CNN-
│
├── cifar10_cnn.py             # Main training script
├── cifar_full_model.pth       # Trained model file (saved using torch.save)
├── README.md                  # This file
```

---

## ⚙️ Requirements

```bash
pip install torch torchvision
```

---

## 🚀 How to Run

### ▶️ Training

```bash
python cifar10_cnn.py
```

This will:
- Download CIFAR-10 dataset
- Normalize using precomputed mean & std
- Train a 4-layer CNN
- Print accuracy & loss per epoch
- Save the model as `cifar_full_model.pth`

---

## 🧠 Model Architecture

```
Input: [3 x 32 x 32]

→ Conv2D(3, 32, 3, padding=1)  
→ ReLU  
→ MaxPool2D(2)  

→ Conv2D(32, 64, 3, padding=1)  
→ ReLU  
→ MaxPool2D(2)  

→ Flatten  
→ Linear(64*8*8, 256)  
→ ReLU  
→ Linear(256, 10)
```

---

## 🧪 Test Sample

Example code to classify a single image from the dataset:

```python
img, label = cifar10[0]
img_tensor = transform(img).unsqueeze(0)

model.eval()
with torch.no_grad():
    output = model(img_tensor)
    predicted = torch.argmax(output, dim=1).item()
```

---

## 📈 Training Output

Example training log:
```
Epoch [1/5], Loss: 1.6843, Accuracy: 0.3976
Epoch [2/5], Loss: 1.3052, Accuracy: 0.5291
...
```

---

## 💾 Saving Model

Model is saved using:

```python
torch.save(model, "cifar_full_model.pth")
```

To load:
```python
model = torch.load("cifar_full_model.pth")
model.eval()
```

---

## 🤝 Contributing

Pull requests are welcome! Feel free to fork the repo and submit your enhancements.

---

## 📩 Contact

Developed by **Zain Jadoon**  
📧 zainjadoon07@gmail.com  
🔗 [GitHub Profile](https://github.com/zainjadoon07)

---

⭐ Star this repo if you find it helpful!
