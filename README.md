# ML-

####### Aufgabe 1 
from torchvision import models
import torch
 
dir(models)

alexnet = models.alexnet(pretrained=True)
alexnet


from PIL import Image
import matplotlib.pyplot as plt

# Erstelle eine Liste mit den Pfaden zu den Bildern
image_paths = ['/Users/FHBBook/Desktop/cat.png', '/Users/FHBBook/Desktop/cat1.png', '/Users/FHBBook/Desktop/cat2.png', '/Users/FHBBook/Desktop/cat3.png', '/Users/FHBBook/Desktop/cat4.png', '/Users/FHBBook/Desktop/cat5.png', '/Users/FHBBook/Desktop/cat6.png', '/Users/FHBBook/Desktop/cat7.png', '/Users/FHBBook/Desktop/cat8.png', '/Users/FHBBook/Desktop/cat10.png']

# Erstelle eine leere Liste für die geöffneten Bilder
#images = []

for path in image_paths:        
    bild = Image.open(path)  # Bild einlesen
    plt.imshow(bild)  # Bild anzeigen
    plt.axis('off')  # Achsen ausschalten
    plt.show()  # Bild anzeige  
    
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as transforms
from IPython.display import display
import matplotlib.pyplot as plt


# Liste von Dateinamen der Bilder
bild_dateinamen = ['/Users/FHBBook/Desktop/cat.png', '/Users/FHBBook/Desktop/cat1.png', '/Users/FHBBook/Desktop/cat2.png', '/Users/FHBBook/Desktop/cat3.png', '/Users/FHBBook/Desktop/cat4.png', '/Users/FHBBook/Desktop/cat5.png', '/Users/FHBBook/Desktop/cat6.png', '/Users/FHBBook/Desktop/cat7.png', '/Users/FHBBook/Desktop/cat8.png', '/Users/FHBBook/Desktop/cat10.png']

# Figur mit 2 Zeilen und 5 Spalten erstellen
fig, axes = plt.subplots(2, 5, figsize=(20, 15))
axes = axes.ravel()

# Schleife, um Bilder einzulesen und zu verarbeiten
for i, bild_dateiname in enumerate(bild_dateinamen):
    # Bild einlesen
    img = Image.open(bild_dateiname)
    
    # Bild in Graustufen konvertieren
    img_gray = img.convert('L')
    
    # Graustufenbild in RGB konvertieren
    img_rgb = img_gray.convert('RGB')
    
    # Transformation durchführen
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img_t = transform(img_rgb)

    # Klassifizierung durchführen
    batch_t = torch.unsqueeze(img_t, 0)
    alexnet.eval()
    out = alexnet(batch_t)

    # Ergebnisse auswerten
    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    results = [(classes[idx], percentage[idx].item()) for idx in indices[0][:11]]
    #print(classes)

    # Bild verkleinern
    img = img.resize((224, 224))

    # Namen und Wahrscheinlichkeiten über das Bild schreiben
    text = '\n'.join([f'{label}: {probability:.2f}%' for label, probability in results])
    
    # Bild in der entsprechenden Zelle des Rasters anzeigen
    axes[i].imshow(img)
    axes[i].axis('off')
    
    
    label = results[0][0]
    probability = results[0][1]
    axes[i].text(0, 0, f'{label}: {probability:.2f}%', transform=axes[i].transAxes,
                 verticalalignment='top', fontsize=17, color='black')  # Textposition unter dem Bild und Textfarbe auf Schwarz setzen

   
    
    # Ergebnisse unter dem Bild schreiben
   

plt.tight_layout()
plt.show()
