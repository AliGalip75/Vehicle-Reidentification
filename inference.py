import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from load_model import load_model_from_opts  # 🔹 doğrudan buradan alıyoruz

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Görüntü dönüşümleri (eğitimle aynı normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Model yükleme
MODEL_DIR = "./model/veri776_resnet50"
model = load_model_from_opts(
    os.path.join(MODEL_DIR, "opts.yaml"),
    os.path.join(MODEL_DIR, "net_19.pth"),
    remove_classifier=True
)
model.to(device).eval()

# --- Tek bir görüntüden feature çıkarımı
def extract_feature(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    feat = model(x)
    feat = F.normalize(feat, p=2, dim=1)
    ff = torch.zeros_like(feat)


    for i in range(2):
        if i == 1:
            x = torch.flip(x, [3])
        with torch.no_grad():
            feat = model(x)
        fnorm = torch.norm(feat, p=2, dim=1, keepdim=True)
        feat = feat.div(fnorm.expand_as(feat))
        ff += feat

    return ff.cpu()


# --- İki görüntü arasındaki benzerlik
def similarity_of_two_images(img1, img2):
    f1 = extract_feature(img1)
    f2 = extract_feature(img2)
    cos = F.cosine_similarity(f1, f2).item()
    return cos

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Kullanım: python inference.py image1.jpg image2.jpg")
        exit(1)

    img1, img2 = sys.argv[1], sys.argv[2]
    cos = similarity_of_two_images(img1, img2)
    print(f"Cosine similarity: {cos:.4f}  ->  similarity %: {(cos + 1) / 2 * 100:.2f}")
