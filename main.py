"""
Tiktok Tarzı Karikatür Filtresi
Öğrenci: [Furkan Yıldırım]
Ders: Görüntü İşleme

Bu program, gerçek zamanlı video akışını karikatür/çizgi film tarzında dönüştüren bir filtre uygulamasıdır.
Temel olarak dört ana görüntü işleme tekniği kullanılmaktadır:
1. Gürültü Azaltma (Bilateral Filtreleme)
2. Kenar Tespiti (Adaptif Eşikleme)
3. Renk Kuantalama (K-means Kümeleme)
4. Renk İyileştirme (HSV Renk Uzayında Doygunluk Artırma)
"""

import cv2
import numpy as np

def apply_vintage_filter(image):
    # Görüntüyü float32 formatına dönüştür
    img_float = image.astype(float) / 255.0
    
    # Sepia efekti için dönüşüm matrisi
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    
    # Sepia dönüşümünü uygula
    sepia_img = cv2.transform(img_float, sepia_matrix)
    
    # Değerleri 0-1 aralığında tut
    sepia_img = np.clip(sepia_img, 0, 1)
    
    # Hafif bir bulanıklık ekle
    sepia_img = cv2.GaussianBlur(sepia_img, (3, 3), 0)
    
    # Kontrastı artır
    sepia_img = np.power(sepia_img, 0.9)
    
    # Vinyetleme efekti ekle
    rows, cols = sepia_img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/4)
    kernel_y = cv2.getGaussianKernel(rows, rows/4)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    
    # Vinyetleme efektini uygula
    for i in range(3):
        sepia_img[:,:,i] = sepia_img[:,:,i] * mask
    
    # Görüntüyü uint8 formatına geri dönüştür
    final_img = (sepia_img * 255).astype(np.uint8)
    return final_img

def apply_cartoon_effect(img):
    """
    Giriş görüntüsüne karikatür efekti uygular.
    
    Parametreler:
    img (numpy.ndarray): BGR formatında giriş görüntüsü
    
    Dönüş:
    numpy.ndarray: Karikatürleştirilmiş görüntü
    """
    
    # Orijinal görüntüyü kopyala
    img_color = img.copy()
    
    # AŞAMA 1: RENK DÜZENLEME
    # Bilateral filtre ile detayları koru
    img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=75, sigmaSpace=75)
    
    # AŞAMA 2: KENAR TESPİTİ
    # Gri tonlamaya çevir
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Median blur uygula
    gray = cv2.medianBlur(gray, 7)
    
    # Kenarları tespit et
    edges = cv2.adaptiveThreshold(gray, 255,
                                cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY,
                                blockSize=9,
                                C=2)
    
    # Kenarları kalınlaştır
    kernel = np.ones((2,2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # AŞAMA 3: RENK SADELEŞTIRME
    # Renk paletini azalt
    div = 64
    img_color = img_color // div * div + div // 2
    
    # AŞAMA 4: KENAR VE RENKLERİ BİRLEŞTİRME
    # Kenarları 3 kanallı yap
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Kenarları ve renkleri birleştir
    cartoon = cv2.bitwise_and(img_color, edges_color)
    
    # AŞAMA 5: RENK İYİLEŞTİRME
    # HSV'ye çevir
    hsv = cv2.cvtColor(cartoon, cv2.COLOR_BGR2HSV)
    
    # Doygunluk ve parlaklığı artır
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 40)  # Doygunluğu artır
    v = cv2.add(v, 20)  # Parlaklığı artır
    
    # Kanalları birleştir
    hsv = cv2.merge([h, s, v])
    
    # BGR'ye geri çevir
    cartoon = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Son rötuşlar
    # Kontrastı artır
    lab = cv2.cvtColor(cartoon, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    cartoon = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return cartoon

def main():
    """
    Ana program döngüsü.
    Webcam'den görüntü alır, filtreyi uygular ve sonucu gösterir.
    Çıkış için 'q' tuşuna basılmalıdır.
    """
    
    # Webcam'i başlat
    cap = cv2.VideoCapture(0)
    
    # Webcam başlatılamadıysa hata ver
    if not cap.isOpened():
        print("Hata: Webcam başlatılamadı!")
        return
    
    # Webcam ayarlarını optimize et
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    while True:
        # Frame'i oku
        ret, frame = cap.read()
        if not ret:
            print("Hata: Frame okunamadı!")
            break
        
        # Karikatür efektini uygula
        cartoon = apply_cartoon_effect(frame)
        
        # Karşılaştırma için orijinal ve filtreli görüntüyü yan yana göster
        combined = np.hstack((frame, cartoon))
        
        # Sonucu ekranda göster
        cv2.imshow('TikTok Karikatür Filtresi (Orijinal | Filtreli)', combined)
        
        # 'q' tuşuna basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Kaynakları serbest bırak
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 