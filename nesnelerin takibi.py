import cv2      #OpenCV kütüphanesi.
import time     #Zamanla ilgili işlemleri yapmak için gereklidir.
#OpenCV ve time kütüphanelerini projeye dahil ettik.

#Videoyu "MOT17-09-DPM-raw.webm" adıyla kaydettim.
#Oluşturduğum ".py" uzantılı Python script dosyama videoyu bir değişken adı ile aşağıdaki komut satırı ile yükledim.

cap = cv2.VideoCapture("MOT17-09-DPM-raw.mp4")

#Kameradan görüntü alma ve video çekme işlemlerinde veya video dosyaları ile yapılacak işlemler için "cv.VideoCapture(*args)" sınıfı kullanılmaktadır.
#Dosyadan video okumak için cv.VideoCapture(*args) komutunu burada kullandım.
#"cap" adında bir değişken oluşturdum ve video çekme işlemini yapmak için VideoCapture komutunu kullandım. 
#Ardından videoyu dahil etmek için video adını ekledim.

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)   

#Arka plan çıkarma işlemi görüntü işleme uygulamalarında sıklıkla kullanılan bir yöntemdir.
#Genellikle sabit bir zemin üzerindeki hareketli nesneleri (insan, araç, ürün vb.) yakalamak ve takip etmek için kullanılır.  
#OpenCV kütüphanesinde arka plan çıkarma işlemi için "BackgroundSubtractorMOG, BackgroundSubtractorMOG 2 ve BackgroundSubtractorGMG" yöntemi geliştirilmiştir.
#Bu nedenle burada cv2.createBackgroundSubtractorMOG2 diyerek arka plan çıkarma işlemi yaptım.

#Threshold eşik değeridir. Giriş olarak verilen görüntüyü ikili görüntüye çevirmek için kullanılan bir yöntemdir.
#İkili görüntü (binary), görüntünün siyah ve beyaz olarak tanımlanmasıdır.
#Threshold, görüntü üzerindeki gürültüleri azaltmak veya nesne belirlemek gibi farklı amaçlar için kullanılır.
#Bu yüzden Threshold'40 değerini verdik. Böylece girdimizi ikili görüntüye çevirmiş oluruz. Ayrıca istenmeyen gürültülerden de kurtulmuş oluruz. 

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape    

#Bu işlem videodan görüntü almak için kullanılır.
#Ret parametresi kameranın çalışıp çalışmadığı ile ilgili true false değeri döndürüyor.
#"Ret", kamera çerçevesini almanın yanlış olduğuna göre geri dönüş değeri elde eder.
#"Frame", kameradaki bir sonraki kareyi elde eder ("cap" ile). 

#cap.read() , bir boolen(True/False) mantık işlemi döndürür. 
#Eğer çerçeve(frame) doğru bir şekilde okunduysa, doğru(True) çıkışını verecektir. Bu bize videonun kontrol edilmesi olanağını verecektir.
#Böylece bu dönüş değerini kontrol ederek videonun sonunu kontrol edebilirim.

#height (yükseklik) ve width(genişliğini) framelerin boyutuna göre ayarladım.


    roi = frame[0: 720,0:1600]     
    
#roi adında bir değişken oluşturdum. Nereden alacağımı yanına ekledim. Frameden alacağım için frame yazdım. 
#x ve y koordinatlarını belirledim. 
#0'dan 720'ye kadar olan aralığı alsın olarak belirledim. Burası y koordinatıdır. 
#0'dan 1600'e kadar olan aralık ise x koordinatıdır. 


    mask = object_detector.apply(roi)   
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   
    
#Mask işlemi renk uzayındaki girilen değerlere denk gelen pikselleri tespit etmede kullanılır ve binary formatta yani 1 ve 0 'lardan oluşan bir görüntü döndürür. 
#Görüntüde ki beyaz yerler nesnemizin bulunduğu alanlar, siyah yerlerse girilen renk koduna uyuşmayan değerlerdir.
#mask adında bir değişken oluşturdum ve daha önceden oluşturmuş olduğum object_detector değişkenine roi değişkeninde yapmış olduğum koordinatları apply ile uygulamış oldum. 

#Konturlar aynı renk ve yoğunluğa sahip olan tüm kesintisiz noktaları sınır boyunca birleştiren bir eğridir. 
#Konturlar şekil analizi,nesne algılama ve tanıma için çok yararlı bir araçtır.
#Kontur bulunması istenirken daha doğru sonuç için binary(siyah-beyaz) formunda resim kullanılmalıdır.
#"FindContours" yöntemiyle konturleri bulunan resim komple değişir orjinal halini bir daha kullanılamaz hale gelir. 
#Burada ilk olarak orjinal resim okunup önce gri skalaya çevirilmiş ve gri skalaya çevrilen resim binary formata getirilmiştir.
#cv2.findContours() işleminde konturlar bulunmuştur.
#cv.RETR_TREE ,tüm konturları alır ve iç içe konturların tam bir hiyerarşisini yeniden oluşturur. 
#Düz bir çizgiden oluşan bir konturu çizdirmek istenirse bütün kontur bilgilerine ihtiyaç yoktur başlangıç ve bitiş noktalarının koordinatlarını bilmek çizdirmek için yeterlidir. 
#Bu durumda ise "cv2.CHAIN_APPROX_SIMPLE" komutunu kullanmak yeterlidir. 

    
    for cnt in contours:
        
        area = cv2.contourArea(cnt)     
        if area > 2000:  
                
#cnt adında bir değişken oluşturdum.
#cv.contourArea() metotu konturun alanını belirlemek için kullanılır. 
#contours listesi çerçevemizde yer alan tüm konturları içerir. Eğer kontur listesi boş değilse, hepsini tek tek işleme sokarız.
#area > 2000 , 2000 piksellik bir alandan küçük konturların etrafına dikdörtgen çizilmemesini sağlamaktır. 
            

            x, y, w, h = cv2.boundingRect(cnt)     
            cv2.rectangle(roi, (x, y), (x + w, y + h), (10, 250, 0), 3)    
#cv2.boundingRect() OpenCV işlevi, ikili görüntünün etrafına yaklaşık bir dikdörtgen çizmek için kullanılır. 
#Bu işlev, esas olarak bir görüntüden konturlar elde ettikten sonra ilgilenilen bölgeyi vurgulamak için kullanılır.
    time.sleep(0.100)
    
#sleep() fonksiyonu, time modülünün en sık kullanılan araçlarından bir tanesidir.
#Bu fonksiyonu kullanarak kodlarımızın işleyişini belli sürelerle kesintiye uğratabiliriz.
#Örnek olarak eğer sleep() fonksiyonuna 0.5 parametresini verirsek, duraklama süresinin 500 milisaniye olmasını sağlarız.
#Burada ise sleep fonksiyonuna 0.001 verdik. Yani saniyede binde bir askıya aldım. 
    
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)     

#Daha önceden oluşturmuş olduğum “frame” ve “mask” değişkenim vardı.
#cv2.imshow metodunun içerisine yazdım. 
#Bu sayede ekranda frame olan videomun adı “frame” mask yani siya beyaz olan videomun adı ise “mask” olmuş olur.


    key = cv2.waitKey(10)
    if key == 27:
        break
    if cv2.waitKey(1) & 0xff == ord("q"):
# "q" tuşuna basıldığında program kapanır.   
        break
    
#Klavye metodu için "cv2.waitKey" kullanılır. Alacağı parametreler milisaniye cinsinden zamandır. Bu fonksiyon herhangi bir klavye olayı için belirtilen süre kadar bekler.
#Herhangi bir şeye basılmadığında program devam eder.
#Son olarak yazdığım cv2.waitKey(n) fonksiyonu, OpenCV de kulanılan keyboard binding fonksiyonudur. 
#Parametre olarak içine bir argüman alır ve bu argüman milisaniye cinsinden süreyi ifade eder. 
#Bu süre boyunca program durur, devam etmek için sürenin bitmesini veya herhangi bir keyboard event’i ile karşılaşmayı bekler.
#Eğer parametre geçirilmezse, 0 değeri (forever demek) default olarak verilir ki bu süre kısıtı olmadan keyboard event dinlemeye yarar.
cap.release()
cv2.destroyAllWindows()

#Yarattığımız pencereler kapatmak için "cv2.destroyAllWindows()" kulllanırız. 
#Döngü kırıldığında kamera nesnesini serbest bırakır ve açılmış tüm pencereleri kapatarak belleği temizleriz.