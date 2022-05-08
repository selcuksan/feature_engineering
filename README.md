# Makine Öğrenmesi için Özellik Mühendisliği

## Bir *Feature* nedir ve bunun mühendisliğine neden ihtiyacımız var? 

  Temel olarak, tüm makine öğrenimi algoritmaları, çıktılar oluşturmak için bazı girdi verilerini kullanır. Bu girdi verileri, genellikle yapılandırılmış sütunlar biçiminde olan özellikleri içerir. 
  Algoritmalar, düzgün çalışması için belirli özelliklere sahip özellikler gerektirir. Burada, özellik mühendisliği ihtiyacı ortaya çıkıyor. 
  Özellik mühendisliği çabalarının temel olarak iki amacı olduğunu söyleyebiliriz:
  
  - Makine öğrenimi algoritması gereksinimleriyle uyumlu, uygun input dataseti'nin hazırlanması.
  - Makine öğrenimi modellerinin performansını iyileştirme.

        Kullandığınız featurelar sonucu her şeyden daha fazla etkiler. Hiçbir algoritma tek başına doğru özellik mühendisliği tarafından sağlanan bilgi kazanımını tamamlayamaz.
        — Luca Massaron
        
*Forbes'taki bir ankete göre, veri bilimcileri zamanlarının %80'ini veri hazırlamaya harcıyor:*

![image](https://user-images.githubusercontent.com/56341239/167311984-5fffd0f1-8e11-4fa8-8bce-2cc162d2bffc.png)

Kaynak: https://www.forbes.com/sites/gilpress/2016/03/23/data-preparation-most-time-consuming-least-enjoyable-data-science-task-survey-says/


## Özellik Mühendisliği Teknikleri

    Imputation
    Handling Outliers
    Binning
    One-Hot Encoding
    Feature Extraction
    Scaling
    
    
### Imputation
  Eksik değerler, verilerinizi makine öğrenimi için hazırlamaya çalıştığınızda karşılaşabileceğiniz en yaygın sorunlardan biridir. 
  Kayıp değerlerin nedeni insan hataları, veri akışındaki kesintiler, gizlilik endişeleri vb. olabilir. 
  Nedeni ne olursa olsun, kayıp değerler makine öğrenmesi modellerinin performansını etkiler.
  
  Bazı makine öğrenme platformları, model eğitim aşamasında eksik değerleri içeren satırları otomatik olarak düşürür ve eğitim boyutunun küçülmesi nedeniyle model performansını düşürür. 
  Öte yandan, algoritmaların çoğu eksik değerlere sahip veri kümelerini kabul etmez ve hata verir.
  
  Eksik değerlere en basit çözüm, satırları veya tüm sütunu silmektir. 
  Silme için optimum bir eşik yoktur ancak örnek değer olarak %70'i kullanabilir ve bu eşiğin üzerindeki eksik değerleri olan satır ve sütunları silmeyi deneyebilirsiniz.
  
   *  Nümerik Imputation


Veri boyutunu koruduğu için Imputation, silmek yerine daha fazla tercih edilen bir seçenektir.
      -   "Geçen aydaki müşteri ziyareti sayısını" gösteren bir sütununuz varsa, eksik değerler 0 ile değiştirilebilir.
      -   Eksik değerlerin bir başka nedeni de farklı boyutlardaki tabloların birleştirilmesidir ve bu durumda 0 değerini almak da mantıklı olabilir.
      -   Eksik değerler için varsayılan bir değere sahip olma durumu dışında, en iyi değerlendirme yolu sütunların medyanlarını kullanmak olabilir. 
      Sütunların ortalamaları aykırı değerlere duyarlı olduğu için medyanlar bu açıdan daha katıdır.
      
   *  Kategorik Imputation


Eksik değerleri mod değeri ile değiştirmek, kategorik sütunları işlemek için iyi bir seçenektir. 

Ancak, sütundaki değerlerin eşit olarak dağıldığını ve baskın bir değer olmadığını düşünüyorsanız, “Diğer” gibi bir kategori yüklemek daha mantıklı olabilir, çünkü böyle bir durumda, tahmininiz rastgele bir seçimi yakınsayabilir.


### Handling Outliers

   *  Standart Sapma ile Aykırı Değer Tespiti

Bir değerin ortalamaya uzaklığı x * standart sapma'dan yüksekse, aykırı değer olarak kabul edilebilir
x için kesin bir çözüm yoktur, ancak genellikle 2 ile 4 arasında bir değer pratik görünmektedir.

Ek olarak bu formül yerine z-skoru da kullanılabilir. Z puanı (veya standart puan), standart sapmayı kullanarak bir değer ile ortalama arasındaki mesafeyi standartlaştırır.

   *  Yüzdelik Değerlerle Aykırı Değer Tespiti

Aykırı değerleri saptamanın başka bir matematiksel yöntemi de yüzdelik dilimleri kullanmaktır. 
Üstten veya alttan gelen değerin belirli bir yüzdesini aykırı değer olarak kabul edebilirsiniz.

### Binning

Binning hem kategorik hem de sayısal verilere uygulanabilir:

* Nümerik Binning Örneği
Değer      Bin       
0-30   ->  Low       
31-70  ->  Mid       
71-100 ->  High

* Kategorik Binning Örneği
  Değer      Bin       
  Spain  ->  Europe      
  Italy  ->  Europe       
  Chile  ->  South America
  Brazil ->  South America
  
 Binning'in ana motivasyonu, modeli daha sağlam hale getirmek ve overfitting'i önlemektir, ancak performansa bir maliyeti vardır. Her binning işleminde, bilgileri feda edersiniz ve verilerinizi daha düzenli hale getirirsiniz.
 
 ### One-hot encoding
 
  One hot encoding, makine öğreniminde en yaygın kodlama yöntemlerinden biridir. Bu yöntem, bir sütundaki değerleri birden çok bayrak sütununa yayar ve bunlara 0 veya 1 atar. Bu ikili değerler, gruplanmış ve kodlanmış sütun arasındaki ilişkiyi ifade eder.
  
  Bu yöntem, algoritmalar için anlaşılması zor olan kategorik verilerinizi sayısal bir formata dönüştürür ve kategorik verilerinizi herhangi bir bilgi kaybı olmadan gruplamanıza olanak tanır.
  
 ![image](https://user-images.githubusercontent.com/56341239/167313041-758c49a6-a9ac-42e0-9394-6f4841db77e7.png)
 

### Feature Extraction
Feature Extraction, mevcut özelliklerin birkaç özelliğe dönüştürülmesini içerir. Başka bir deyişle, özellik çıkarma, mevcut özellikleri birleştirerek bir özellik alt kümesi oluşturmaktır. 
Özellik çıkarmanın bir dezavantajı, oluşturulan yeni özelliklerin insanlar tarafından yorumlanamamasıdır. Yeni değişkenlerdeki veriler, insan gözüne rastgele sayılar gibi görünecektir.

![image](https://user-images.githubusercontent.com/56341239/167313415-3015e046-f2da-413c-97fb-eeff7f02a72b.png)

### Scaling

Çoğu durumda, veri kümesinin sayısal özellikleri belirli bir aralığa sahip değildir ve birbirlerinden farklılık gösterirler. Gerçek hayatta, yaş ve gelir sütunlarının aynı aralığa sahip olmasını beklemek saçmadır. Ancak makine öğrenimi açısından bu iki sütun nasıl karşılaştırılabilir?

Scaling bu sorunu çözer. Bir Scaling işleminden sonra sürekli özellikler, aralık açısından özdeş hale gelir. Bu işlem birçok algoritma için zorunlu değildir.
Ancak, k-NN veya k-Means gibi mesafe hesaplamalarına dayalı algoritmaların, model girişi olarak ölçeklenmiş sürekli özelliklere sahip olması gerekir.

Normalization

![image](https://user-images.githubusercontent.com/56341239/167313508-ebee4f74-60b0-4e60-8759-e57f2cb8eea8.png)

Standardization

![image](https://user-images.githubusercontent.com/56341239/167313518-5fbfb1a9-a84b-4977-841d-89c95febaac8.png)

