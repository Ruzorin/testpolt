import * as tf from "@tensorflow/tfjs";
import * as use from "@tensorflow-models/universal-sentence-encoder";

// Kategorilendirme modeli
export class ContentAnalyzer {
  private model: tf.LayersModel | null = null;
  private encoder: any;

  constructor() {
    this.initializeModel();
  }

  private async initializeModel() {
    // Universal Sentence Encoder yükle
    this.encoder = await use.load();

    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ units: 128, activation: "relu", inputShape: [512] })); // USE çıkış boyutu
    this.model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    this.model.add(tf.layers.dense({ units: 5, activation: "softmax" })); // 5 kategori için çıkış
    this.model.compile({ optimizer: "adam", loss: "categoricalCrossentropy", metrics: ["accuracy"] });
  }

  async analyzeContent(content: string): Promise<string[]> {
    if (!this.model || !this.encoder) throw new Error("Model veya encoder başlatılmadı");

    // Metni vektörleştir (Universal Sentence Encoder ile)
    const embeddings = await this.encoder.embed([content]);
    const input = embeddings.arraySync()[0] as number[];

    const inputTensor = tf.tensor2d([input], [1, 512]);
    const prediction = this.model.predict(inputTensor) as tf.Tensor;
    const scores = prediction.dataSync() as Float32Array;

    const categories = ["Bilim", "Spor", "Siyaset", "Eğlence", "Futbol"];
    const result: string[] = [];
    for (let i = 0; i < scores.length; i++) {
      if (scores[i] > 0.4) result.push(categories[i]); // Daha düşük eşik
    }

    return result.length ? result : ["Genel"];
  }

  // Modeli eğit (örnek veriyle)
  async trainContentModel(trainingData: { text: string; categories: string[] }[]) {
    if (!this.model || !this.encoder) throw new Error("Model veya encoder başlatılmadı");

    const inputs = [];
    const labels = [];
    for (const data of trainingData) {
      const embeddings = await this.encoder.embed([data.text]);
      const input = embeddings.arraySync()[0] as number[];
      inputs.push(input);

      const labelVector = new Array(5).fill(0);
      data.categories.forEach((cat) => {
        const index = categories.indexOf(cat);
        if (index !== -1) labelVector[index] = 1;
      });
      labels.push(labelVector);
    }

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 512]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 5]);

    await this.model.fit(inputTensor, labelTensor, {
      epochs: 100,
      batchSize: 32,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}, accuracy = ${log.acc}`),
      },
    });

    await this.model.save("file://./contentModel");
    console.log("İçerik kategorilendirme modeli eğitimi tamamlandı ve kaydedildi.");
  }
}

// Örnek kullanım
const contentAnalyzer = new ContentAnalyzer();
contentAnalyzer.analyzeContent("Bilimsel bir keşif yaptık!").then(console.log);