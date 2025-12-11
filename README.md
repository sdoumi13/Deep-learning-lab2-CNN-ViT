# Deep-learning-lab2-CNN-ViT
# Deep Learning Lab 2 - Comparaison d'architectures sur MNIST

**Auteur :** [Votre Nom]  
**Sujet :** Exploration de CNN classique, Faster R-CNN et Transfer Learning sur le dataset MNIST  

## 📌 Introduction

L’objectif de ce laboratoire était d’explorer et de comparer différentes approches architecturales pour la classification d’images sur le dataset MNIST :

- Un CNN classique (baseline)
- Un modèle de détection d’objets Faster R-CNN détourné pour faire de la classification
- Des approches de Transfer Learning avec VGG16 et AlexNet

Ce README résume la démarche, les implémentations clés et l’analyse des résultats obtenus.

---

## 1. CNN Classique (Baseline)

### 🧠 Approche
Architecture légère avec extraction de caractéristiques locales (convolutions) + réduction de dimensionnalité (max pooling).

### 💻 Structure principale
```python
self.conv_layers = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2)
)
📊 Résultats





















MétriqueValeurAccuracy (test)99.01 %Temps d’entraînement73.5 s (5 époques)Loss finale0.032
Observation : Convergence extrêmement rapide. Sur MNIST (images 28×28 centrées), un CNN léger atteint presque la perfection.

2. Faster R-CNN (Détection d’objet → Classification)
🧠 Approche
Utilisation d’un détecteur d’objets (backbone ResNet50 + FPN + RPN) en considérant chaque chiffre comme un unique « objet » à localiser.
💻 Point clé : création de bounding boxes automatiques
Pythonnon_zero = torch.nonzero(img_tensor.squeeze())
x_min, x_max = torch.min(non_zero[:, 1]).item(), torch.max(non_zero[:, 1]).item()
y_min, y_max = torch.min(non_zero[:, 0]).item(), torch.max(non_zero[:, 0]).item()

target["boxes"] = torch.as_tensor([[x_min, y_min, x_max+1, y_max+1]], dtype=torch.float32)
target["labels"] = torch.as_tensor([label + 1], dtype=torch.int64)  # +1 car 0 = background
📊 Résultats

















MétriqueValeurAccuracy (test)94.80 %Temps d’entraînement778 s (~13 min)
Observation critique : ×10 plus lent que le CNN simple, et moins précis.
Verdict : Totalement overkill pour une tâche où la localisation est triviale.

3. Transfer Learning (VGG16 & AlexNet)
🧠 Approche
Modèles pré-entraînés ImageNet → adaptation à MNIST (grayscale 28×28 → RGB 224×224).
💻 Transformations & freeze
Pythontransform_tl = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

# Freeze des couches convolutionnelles
for param in vgg16.parameters():
    param.requires_grad = False

# Remplacement de la tête
num_ftrs = vgg16.classifier[6].in_features
vgg16.classifier[6] = nn.Linear(num_ftrs, 10)
📊 Résultats (loss sur 5 époques)

















ModèleLoss initiale → finaleVGG161.36 → 0.74AlexNet0.94 → 0.49
Observation : Entraînement très lourd à cause de l’upscaling ×64 des images. Gain de performance négligeable vs CNN natif.

🏆 Synthèse Globale





























ModèleAccuracyTemps (5 époques)VerdictCNN Standard99.01 %73.5 s✅ Optimal – meilleur ratio perf/coûtFaster R-CNN94.80 %778 s❌ Inadapté – trop complexeTransfer Learning~98-99 %Très élevé⚠️ Coûteux – upscaling pénalisant

🎯 Conclusion du Laboratoire
La complexité d’un modèle ne garantit jamais de meilleures performances.
Sur un dataset simple et bien structuré comme MNIST :
Un CNN léger et dédié surpasse largement largement des architectures massifs (Faster R-CNN, VGG16 pré-entraîné) tant en précision qu’en vitesse.
Ce lab illustre parfaitement le principe : "Choose the right tool for the job".