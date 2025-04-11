import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet";
import * as cocoSsd from "@tensorflow-models/coco-ssd";

// Görüntü analizi modeli
export class ImageAnalyzer {
  private mobilenetModel: mobilenet.MobileNet | null = null;
  private cocoSsdModel: cocoSsd.ObjectDetection | null = null;

  constructor() {
    this.initializeModels();
  }

  private async initializeModels() {
    this.mobilenetModel = await mobilenet.load();
    this.cocoSsdModel = await cocoSsd.load();
  }

  async analyzeImage(imageUrl: string): Promise<{
    categories: string[];
    objects: { name: string; score: number; bbox: [number, number, number, number] }[];
    isAppropriate: boolean;
  }> {
    if (!this.mobilenetModel || !this.cocoSsdModel) throw new Error("Modeller başlatılmadı");

    // Görüntüyü yükle ve Tensor’a dönüştür
    const img = new Image();
    img.crossOrigin = "Anonymous";
    img.src = imageUrl;
    await new Promise((resolve) => (img.onload = resolve));

    const tensor = tf.browser.fromPixels(img).resizeNearestNeighbor([224, 224]).toFloat().div(tf.scalar(255)) as tf.Tensor3D;

    // Kategorilendirme (MobileNet ile)
    const predictions = await this.mobilenetModel.classify(tensor);
    const categories = predictions.map((p) => p.className).slice(0, 5); // En iyi 5 kategori

    // Nesne tespiti (Coco SSD ile)
    const objects = await this.cocoSsdModel.detect(img);
    const detectedObjects = objects.map((obj) => ({
      name: obj.class,
      score: obj.score,
      bbox: [obj.bbox.xmin, obj.bbox.ymin, obj.bbox.width, obj.bbox.height],
    }));

    // Uygunluk kontrolü (örnek kurallar)
    const inappropriateObjects = ["knife", "gun", "violence", "nude"];
    const isAppropriate = !detectedObjects.some((obj) => inappropriateObjects.includes(obj.name.toLowerCase()));

    tensor.dispose(); // Bellek temizliği
    return { categories, objects: detectedObjects, isAppropriate };
  }

  // Görüntü verisiyle modeli eğit (varsa)
  async trainImageModel(trainingData: { imageUrl: string; categories: string[]; isAppropriate: boolean }[]) {
    // MobileNet ve Coco SSD önceden eğitilmiş modeller olduğu için ek eğitim genelde gerekmez
    // Ancak, özel bir model eklenebilir
    const customModel = tf.sequential();
    customModel.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: "relu", inputShape: [224, 224, 3] }));
    customModel.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
    customModel.add(tf.layers.flatten());
    customModel.add(tf.layers.dense({ units: 64, activation: "relu" }));
    customModel.add(tf.layers.dense({ units: 5, activation: "softmax" })); // Kategori sayısı
    customModel.compile({ optimizer: "adam", loss: "categoricalCrossentropy", metrics: ["accuracy"] });

    const inputs = [];
    const labels = [];
    for (const data of trainingData) {
      const img = new Image();
      img.crossOrigin = "Anonymous";
      img.src = data.imageUrl;
      await new Promise((resolve) => (img.onload = resolve));
      const tensor = tf.browser.fromPixels(img).resizeNearestNeighbor([224, 224]).toFloat().div(tf.scalar(255)) as tf.Tensor3D;
      inputs.push(tensor.dataSync());

      const labelVector = new Array(5).fill(0);
      data.categories.forEach((cat) => {
        const index = ["Bilim", "Spor", "Siyaset", "Eğlence", "Diğer"].indexOf(cat);
        if (index !== -1) labelVector[index] = 1;
      });
      labels.push(labelVector);
    }

    const inputTensor = tf.tensor3d(inputs, [inputs.length, 224, 224, 3]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 5]);

    await customModel.fit(inputTensor, labelTensor, {
      epochs: 100,
      batchSize: 16,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}, accuracy = ${log.acc}`),
      },
    });

    await customModel.save("file://./imageModel");
    console.log("Görüntü analizi modeli eğitimi tamamlandı ve kaydedildi.");
  }
}

// Örnek kullanım
const imageAnalyzer = new ImageAnalyzer();
imageAnalyzer.analyzeImage("https://example.com/image.jpg").then(console.log);