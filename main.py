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

import cv2  # OpenCV kütüphanesi, görüntü işleme ve bilgisayarla görme için kullanılır.
import numpy as np  # NumPy kütüphanesi, matematiksel işlemler ve dizi manipülasyonu için kullanılır.
import time  # Zaman işlemleri için kullanılır, özellikle geri sayım için.
import tkinter as tk  # Tkinter, GUI (grafik kullanıcı arayüzü) oluşturmak için kullanılır.
from tkinter import messagebox  # Tkinter'dan mesaj kutuları için kullanılır.
from PIL import Image, ImageTk  # PIL (Python Imaging Library), görüntü işleme ve görüntü formatları arasında dönüşüm için kullanılır.

# Global frame değişkeni
frame = None
cap = None  # Webcam nesnesi

def apply_vintage_filter(image):
    # Görüntüyü float32 formatına dönüştür
    img_float = image.astype(float) / 255.0  # Görüntüyü 0-1 aralığına normalize et
    
    # Sepia efekti için dönüşüm matrisi
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    
    # Sepia dönüşümünü uygula
    sepia_img = cv2.transform(img_float, sepia_matrix)  # Sepia efekti uygula
    
    # Değerleri 0-1 aralığında tut
    sepia_img = np.clip(sepia_img, 0, 1)  # Değerleri sınırla
    
    # Hafif bir bulanıklık ekle
    sepia_img = cv2.GaussianBlur(sepia_img, (3, 3), 0)  # Görüntüyü bulanıklaştır
    
    # Kontrastı artır
    sepia_img = np.power(sepia_img, 0.9)  # Kontrastı artır
    
    # Vinyetleme efekti ekle
    rows, cols = sepia_img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/4)  # Yatay vinyetleme için Gaussian kernel
    kernel_y = cv2.getGaussianKernel(rows, rows/4)  # Dikey vinyetleme için Gaussian kernel
    kernel = kernel_y * kernel_x.T  # Vinyetleme maskesi oluştur
    mask = kernel / kernel.max()  # Maskeyi normalize et
    
    # Vinyetleme efektini uygula
    for i in range(3):
        sepia_img[:,:,i] = sepia_img[:,:,i] * mask  # Her kanala vinyetleme uygula
    
    # Görüntüyü uint8 formatına geri dönüştür
    final_img = (sepia_img * 255).astype(np.uint8)  # Görüntüyü 0-255 aralığına döndür
    return final_img

def apply_cartoon_effect(img):
    # AŞAMA 1: GÜRÜLTÜ AZALTMA
    img_blur = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)  # Gürültüyü azaltmak için bilateral filtre uygula
    
    # AŞAMA 2: KENAR TESPİTİ
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)  # Renkli görüntüyü gri tonlamaya çevir
    edges = cv2.adaptiveThreshold(gray, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,
                                blockSize=5,
                                C=3)  # Kenar tespiti için adaptif eşikleme uygula
    
    # AŞAMA 3: RENK İŞLEME
    img_color = img_blur.copy()  # Bulanık görüntüyü kopyala
    
    # Gölge tespiti için gri tonlama
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    # Gölge maskesi oluştur
    _, shadow_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)  # Gölge maskesi oluştur
    shadow_mask = cv2.bitwise_not(shadow_mask)  # Gölge maskesini ters çevir
    
    # Renk kuantalama (gölgeler hariç)
    img_color = np.uint8(np.round(img_color / 15.0) * 15.0)  # Renk kuantalama uygula
    
    # HSV uzayına çevir
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)  # BGR'den HSV'ye çevir
    h, s, v = cv2.split(img_hsv)  # HSV bileşenlerini ayır
    
    # Gölgeli olmayan alanlarda doygunluğu artır
    s_boost = np.where(
        shadow_mask == 0,
        np.clip(s * 1.2, 0, 255),
        np.clip(s * 0.8, 0, 255)
    ).astype(np.uint8)  # Doygunluğu artır
    
    # Parlaklık ayarı (gölgelere göre adaptif)
    v_boost = np.where(
        shadow_mask == 0,
        np.clip(v * 1.2, 0, 255),
        np.clip(v * 0.8, 0, 255)
    ).astype(np.uint8)  # Parlaklığı artır
    
    # Renkleri birleştir
    img_hsv = cv2.merge([h, s_boost, v_boost])  # HSV bileşenlerini birleştir
    img_color = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)  # HSV'den BGR'ye çevir
    
    # Son rötuş için hafif yumuşatma
    img_color = cv2.GaussianBlur(img_color, (1, 1), 0)  # Daha az bulanıklık için boyutu 1x1 yaptık
    
    # AŞAMA 4: KENAR VE RENKLERİ BİRLEŞTİRME
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Kenarları renkli formata çevir
    cartoon = cv2.bitwise_and(img_color, img_color, mask=edges)  # Kenar maskesi ile renkli görüntüyü birleştir
    
    return cartoon  # Orijinal boyutta döndür

def capture_photo():
    global frame
    # Karikatür efekti uygulanmış görüntüyü al
    cartoon_frame = apply_cartoon_effect(frame)  # Karikatür efekti uygula
    # Fotoğrafı yüksek çözünürlükte kaydet
    cv2.imwrite('captured_photo.png', cartoon_frame)  # Karikatür efekti uygulanmış frame'i kaydet
    print("Fotoğraf kaydedildi: captured_photo.png")
    messagebox.showinfo("Fotoğraf Çekildi", "Fotoğraf kaydedildi: captured_photo.png")  # Kullanıcıya bilgi ver

def update_frame():
    global frame
    ret, frame = cap.read()  # Webcam'den bir frame oku
    if ret:
        cartoon = apply_cartoon_effect(frame)  # Frame'e karikatür efekti uygula
        
        # OpenCV görüntüsünü tkinter etiketine yerleştir
        img = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)  # BGR'den RGB'ye çevir
        img = Image.fromarray(img)  # NumPy dizisini PIL görüntüsüne çevir
        img = ImageTk.PhotoImage(img)  # PIL görüntüsünü PhotoImage'e çevir

        # Label'a resmi yerleştir
        preview_label.imgtk = img
        preview_label.configure(image=img)  # Etiketi güncelle

    preview_label.after(10, update_frame)  # Her 10 ms'de bir frame güncelle

def start_capture():
    global cap
    cap = cv2.VideoCapture(0)  # Webcam'i başlat
    
    if not cap.isOpened():
        print("Hata: Webcam başlatılamadı!")  # Webcam açılmazsa hata mesajı ver
        return

    update_frame()  # İlk frame'i güncelleyerek başlat

def on_capture_button_click():
    print("Fotoğraf çekiliyor...")
    countdown_label = tk.Label(root, text="", font=("Helvetica", 48))  # Geri sayım için etiket
    countdown_label.pack(pady=20)

    for i in range(3, 0, -1):
        countdown_label.config(text=str(i))  # Geri sayımı güncelle
        root.update()  # Tkinter arayüzünü güncelley
        time.sleep(1)  # 1 saniye bekle

    countdown_label.pack_forget()  # Geri sayım etiketini kaldır
    capture_photo()  # Fotoğrafı çek

if __name__ == "__main__":
    # Tkinter arayüzü oluştur
    root = tk.Tk()
    root.title("Karikatür Filtresi")  # Pencere başlığı

    # Pencere boyutunu ayarla
    root.geometry("720x720") 

    # Önizleme paneli
    preview_label = tk.Label(root, width=720, height=480)  # Önizleme için etiket
    preview_label.pack(pady=10)

    # Fotoğraf çek butonu
    capture_button = tk.Button(root, text="Fotoğraf Çek", command=on_capture_button_click, width=20)  # Buton oluştur
    capture_button.pack(pady=10)

    # Uygulamayı başlat
    start_capture()    
    root.mainloop()  # Tkinter döngüsünü başlat
