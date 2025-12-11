# Rapport de Laboratoire : Architectures Deep Learning (CNN, R-CNN, ViT)

**Sujet :** Deep Learning Lab 2 - Comparaison d'architectures sur MNIST  
**Auteur :** [Votre Nom / Groupe]  
**Contexte :** Exploration des paradigmes de Convolution, Détection d'Objet et Mécanismes d'Attention.

---

## 📌 1. Introduction & Objectifs
L'objectif de ce laboratoire est d'analyser le comportement de différentes architectures de réseaux de neurones sur un problème standard (MNIST). Nous avons cherché à comprendre comment des architectures radicalement différentes (CNN classique, Détecteur d'objets, Transformer) abordent la même tâche de classification et quels sont leurs coûts respectifs en termes de calcul et de performance.

---

## 🏛️ Partie 1 : Approches Convolutionnelles & Détection

### 1.1 Le CNN Standard (Baseline)
**Logique Théorique :** Le CNN (Convolutional Neural Network) est l'architecture naturelle pour le traitement d'images. Il utilise l'invariance par translation via des filtres locaux (convolutions) pour extraire des caractéristiques hiérarchiques (bords -> formes -> chiffres).

**Implémentation :** J'ai conçu un modèle léger ("from scratch") alternant extraction de features et réduction de dimensionnalité.

```python
# Extrait de mon architecture CNN
self.conv_layers = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1), # Feature Map: 28x28
    nn.ReLU(),
    nn.MaxPool2d(2),                            # Downsampling: 14x14
    nn.Conv2d(32, 64, kernel_size=3, padding=1) # Augmentation de la profondeur
)
```

**Résultats :** Accuracy : 99.01%  
**Temps :** ~73s

**Analyse :** Convergence extrêmement rapide. Le modèle "sait" naturellement comment traiter l'image grâce à l'inductance biaisée des convolutions.

---

### 1.2 Faster R-CNN (Le Défi Technique)
**Logique Théorique :** Faster R-CNN est conçu pour la détection (trouver où est l'objet et ce que c'est). Pour l'appliquer à MNIST, j'ai dû formuler l'hypothèse que chaque chiffre est un "objet" à localiser, même si l'image est centrée.

**Implémentation (Le "Hack") :** Le modèle nécessite des coordonnées de boîtes (x1, y1, x2, y2). J'ai créé un Dataset personnalisé qui génère dynamiquement ces boîtes en détectant les pixels non-nuls du chiffre.

```python
# Génération dynamique des Bounding Boxes dans le Dataset
non_zero = torch.nonzero(img_tensor.squeeze())
x_min, x_max = torch.min(non_zero[:, 1]).item(), torch.max(non_zero[:, 1]).item()
# La cible devient une boîte englobante + le label
target["boxes"] = torch.as_tensor([[x_min, y_min, x_max+1, y_max+1]], dtype=torch.float32)
```

**Résultats :** Accuracy : 94.80%  
**Temps :** ~778s (Env. 13 min)

**Analyse :** Le modèle est 10x plus lent que le CNN. C'est une architecture "Overkill" : le réseau perd énormément de ressources à proposer des régions (RPN) pour localiser un objet qui est toujours au centre.

---

### 1.3 Transfer Learning (VGG16 & AlexNet)
**Logique Théorique :** Utilisation de modèles très profonds pré-entraînés sur ImageNet. La contrainte majeure est l'adaptation dimensionnelle (ImageNet = 224x224 RGB vs MNIST = 28x28 Gris).

**Implémentation :** J'ai dû upscaler artificiellement les images, ce qui augmente drastiquement la mémoire requise.

```python
transform_tl = transforms.Compose([
    transforms.Resize((224, 224)),       # Upscaling x8
    transforms.Grayscale(num_output_channels=3), # Adaptation RGB
    transforms.ToTensor()
])
# Freeze des poids pour ne ré-entraîner que la couche finale
for param in vgg16.parameters(): param.requires_grad = False
```

**Analyse :** Bien que fonctionnelle, cette approche est inefficace pour MNIST car l'upscaling crée une redondance de données massive (64x plus de pixels à traiter).

---

## 👁️ Partie 2 : Vision Transformer (ViT)

### 2.1 Approche "Attention Is All You Need"
**Logique Théorique :** Contrairement au CNN qui regarde les pixels voisins, le ViT découpe l'image en "patches" (carrés) et utilise le mécanisme de Self-Attention pour que chaque patch puisse "voir" tous les autres patches instantanément. C'est une approche globale et non locale.

### Implémentation "From Scratch"
J'ai implémenté le découpage en patches et l'ajout d'embeddings positionnels (car l'attention n'a pas de notion d'ordre spatial).

```python
# Découpage de l'image en patches via Convolution
self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=7, stride=7)
# Ajout de l'information de position (apprise)
x = x + self.pos_embed
```

**Résultats :** Accuracy : ~97% - 98%

**Analyse :** Le ViT performe étonnamment bien pour une implémentation "from scratch". Cependant, il est généralement moins performant que le CNN sur de petits datasets car il manque de "Biais Inductif" (il doit apprendre que les pixels voisins sont corrélés, alors que le CNN le sait par design).

---

## 📊 Partie 3 : Analyse Graphique & Synthèse

### 3.1 Comparaison de Performance (Accuracy / F1)
Les résultats montrent :
- Le **CNN Standard domine** légèrement (~99%).
- Le **Faster R-CNN** (~94.8%) souffre de sa complexité inutile.
- Le **ViT** (~97-98%) est performant mais demande plus de données.

### 3.2 Comparaison Temporelle
Les temps d'entraînement montrent une disparité massive :
- **CNN : ~73s**
- **Faster R-CNN : ~778s** (10× plus lent)
- **ViT : entre les deux**

**Interprétation :** Le coût computationnel du R-CNN (RPN, RoI Align, etc.) est injustifiable pour de la simple classification.

---

## 🏆 Conclusion Générale
Ce laboratoire démontre que **la complexité n'est pas toujours synonyme de performance**.

- Pour des tâches simples (images centrées, faible résolution) : **le CNN est roi**.
- Pour la détection d'objets multiples : **Faster R-CNN reste nécessaire**, malgré son coût.
- Pour de grands datasets avec relations globales : **le ViT est l'état de l'art**, mais il est data-hungry.

Ce travail a permis de valider expérimentalement les théories de coût/bénéfice des architectures modernes en Deep Learning.

