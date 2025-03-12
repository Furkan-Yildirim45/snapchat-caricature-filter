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
import time
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Global frame değişkeni
frame = None
cap = None  # Webcam nesnesi

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
    # AŞAMA 1: GÜRÜLTÜ AZALTMA
    img_blur = cv2.medianBlur(img, 5)
    
    # AŞAMA 2: KENAR TESPİTİ
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,
                                blockSize=5,
                                C=2)
    
    # AŞAMA 3: RENK İŞLEME
    img_color = img_blur.copy()
    
    # Gölge tespiti için gri tonlama
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    # Gölge maskesi oluştur
    _, shadow_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    shadow_mask = cv2.bitwise_not(shadow_mask)
    
    # Renk kuantalama (gölgeler hariç)
    img_color = np.uint8(np.round(img_color / 15.0) * 15.0)
    
    # HSV uzayına çevir
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    
    # Gölgeli olmayan alanlarda doygunluğu artır
    s_boost = np.where(
        shadow_mask == 0,
        np.clip(s * 1 + 10, 0, 255),
        np.clip(s * 0.7, 0, 255)
    ).astype(np.uint8)
    
    # Parlaklık ayarı (gölgelere göre adaptif)
    v_boost = np.where(
        shadow_mask == 0,
        np.clip(v * 1.1, 0, 255),
        np.clip(v * 0.7, 0, 255)
    ).astype(np.uint8)
    
    # Renkleri birleştir
    img_hsv = cv2.merge([h, s_boost, v_boost])
    img_color = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    
    # Son rötuş için hafif yumuşatma
    img_color = cv2.GaussianBlur(img_color, (3, 3), 0)
    
    # AŞAMA 4: KENAR VE RENKLERİ BİRLEŞTİRME
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(img_color, img_color, mask=edges)
    
    return cartoon  # Orijinal boyutta döndür

def capture_photo():
    global frame
    # Karikatür efekti uygulanmış görüntüyü al
    cartoon_frame = apply_cartoon_effect(frame)
    # Fotoğrafı yüksek çözünürlükte kaydet
    cv2.imwrite('captured_photo.png', cartoon_frame)  # Karikatür efekti uygulanmış frame'i kaydet
    print("Fotoğraf kaydedildi: captured_photo.png")
    messagebox.showinfo("Fotoğraf Çekildi", "Fotoğraf kaydedildi: captured_photo.png")

def update_frame():
    global frame
    ret, frame = cap.read()
    if ret:
        cartoon = apply_cartoon_effect(frame)
        
        # OpenCV görüntüsünü tkinter etiketine yerleştir
        img = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)  # BGR'den RGB'ye çevir
        img = Image.fromarray(img)  # NumPy dizisini PIL görüntüsüne çevir
        img = ImageTk.PhotoImage(img)  # PIL görüntüsünü PhotoImage'e çevir

        # Label'a resmi yerleştir
        preview_label.imgtk = img
        preview_label.configure(image=img)

    preview_label.after(10, update_frame)  # Her 10 ms'de bir frame güncelle

def start_capture():
    global cap
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Hata: Webcam başlatılamadı!")
        return

    update_frame()  # İlk frame'i güncelleyerek başlat

def on_capture_button_click():
    print("Fotoğraf çekiliyor...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    capture_photo()  # Fotoğrafı çek

if __name__ == "__main__":
    # Tkinter arayüzü oluştur
    root = tk.Tk()
    root.title("Karikatür Filtresi")

    # Pencere boyutunu ayarla
    root.geometry("720x600")  # 720 genişlik, 600 yükseklik

    # Önizleme paneli
    preview_label = tk.Label(root, width=720, height=480)
    preview_label.pack(pady=10)

    # Fotoğraf çek butonu
    capture_button = tk.Button(root, text="Fotoğraf Çek", command=on_capture_button_click, width=20)
    capture_button.pack(pady=10)

    # Uygulamayı başlat
    start_capture()    
    root.mainloop()
