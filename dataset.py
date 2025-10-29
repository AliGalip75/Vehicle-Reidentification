from torch.utils.data import Dataset
from PIL import Image
import random
import os

class ImageDataset(Dataset):
    """Görüntülerin ve etiketlerin DataFrame formatinda tutulduğu etiketli bir veri kümesi sinifi."""

    def __init__(self, img_root, df, target_label, classes="infer", transform=None, target_transform=None):
        # Görüntülerin bulunduğu klasör yolu
        self.img_root = img_root
        # Görüntü bilgilerini içeren DataFrame
        self.df = df
        # Etiket sütunu adı
        self.target_label = target_label
        # Görüntülere uygulanacak dönüşümler
        self.transform = transform
        # Etiketlere uygulanacak dönüşümler
        self.target_transform = target_transform

        # Sınıf etiketleri otomatik çıkarılacaksa
        if classes == "infer":
            self.classes = list(set(df[target_label].values))
        else:
            self.classes = classes
            class_set = set(df[target_label].values)
            assert class_set.issubset(set(classes)), "Hata: Veri çerçevesindeki sınıflar sağlanan sınıfların alt kümesi olmalıdır."

        # Sınıf adlarını sayısal etiketlere eşleyen sözlük
        self.class_idx = {cl: idx for idx, cl in enumerate(self.classes)}

    def __len__(self):
        # Veri kümesindeki örnek sayısını döndür
        return len(self.df)

    def get_image(self, idx):
        """
        Belirli bir indeksteki görüntüyü yükler ve RGB formatina dönüştürür.
        Bu metod ham görüntüyü almak için kullanılır.
        """
        row = self.df.loc[idx]
        pth = os.path.join(self.img_root, row["path"])
        image = Image.open(pth).convert("RGB")
        return image

    def __getitem__(self, idx):
        """
        Belirli bir indeksteki (image, label) çiftini döndürür.
        Transform işlemleri varsa uygular.
        """
        row = self.df.loc[idx]
        pth = os.path.join(self.img_root, row["path"])
        image = Image.open(pth).convert("RGB")
        label = self.class_idx[row[self.target_label]]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class BatchSampler:
    def __init__(self, dataset, batch_size, samples_per_class, drop_last=True):
        """
        Belirtilen sayıda sınıf başına örnekle, veri kümesinden örnekleri gruplar halinde örnekleyen sınıf.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.drop_last = drop_last
        self.batches, self.batch_idx = [], 0

    def __iter__(self):
        # Etiketleri karıştır
        ids = self.dataset.df[self.dataset.target_label]
        ids = ids.sample(frac=1.0)
        samples_for_id = {}

        # Her sınıf için indeksleri grupla
        for idx, cls in ids.items():
            samples_for_id.setdefault(cls, []).append(idx)

        # Belirtilen sayıda örnek içeren yığınlar (patches) oluştur
        patches = []
        for _, samples in samples_for_id.items():
            for i in range(0, len(samples), self.samples_per_class):
                patches.append(samples[i:i + self.samples_per_class])

        # Yığınları karıştır
        random.shuffle(patches)
        self.batches, self.batch_idx = [[]], 0

        # Yığınları batch_size'a göre grupla
        for patch in patches:
            last_batch = self.batches[-1]
            if len(patch) + len(last_batch) <= self.batch_size:
                last_batch.extend(patch)
            else:
                num_needed = self.batch_size - len(last_batch)
                last_batch.extend(patch[:num_needed])
                self.batches.append(patch[num_needed:])

        # Son batch eksikse ve drop_last True ise çıkar
        if len(self.batches[-1]) < self.batch_size:
            self.batches.pop()
        return self

    def __len__(self):
        # Toplam batch sayısını döndür
        n_samples = len(self.dataset.df)
        n_batches = n_samples // self.batch_size
        if not self.drop_last and n_samples % self.batch_size != 0:
            n_batches += 1
        return n_batches

    def __next__(self):
        # Bir sonraki batch'i döndür
        self.batch_idx += 1
        if self.batch_idx > len(self.batches):
            raise StopIteration()
        return self.batches[self.batch_idx - 1]
