import * as tf from "@tensorflow/tfjs";
import * as use from "@tensorflow-models/universal-sentence-encoder";

// İçerik moderasyon modeli
export class ContentModerator {
  private model: tf.LayersModel | null = null;
  private encoder: any;

  constructor() {
    this.initializeModel();
  }

  private async initializeModel() {
    // Universal Sentence Encoder yükle
    this.encoder = await use.load();

    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ units: 256, activation: "relu", inputShape: [512] }));
    this.model.add(tf.layers.dropout({ rate: 0.3 }));
    this.model.add(tf.layers.dense({ units: 128, activation: "relu" }));
    this.model.add(tf.layers.dense({ units: 2, activation: "softmax" })); // Uygun (0) veya Uygunsuz (1)
    this.model.compile({ optimizer: "adam", loss: "categoricalCrossentropy", metrics: ["accuracy"] });
  }

  async moderateContent(content: string): Promise<{ isAppropriate: boolean; score: number; reason?: string }> {
    if (!this.model || !this.encoder) throw new Error("Model veya encoder başlatılmadı");

    // Metni vektörleştir
    const embeddings = await this.encoder.embed([content]);
    const input = embeddings.arraySync()[0] as number[];

    const inputTensor = tf.tensor2d([input], [1, 512]);
    const prediction = this.model.predict(inputTensor) as tf.Tensor;
    const scores = prediction.dataSync() as Float32Array;

    const isAppropriate = scores[0] > 0.6; // Uygun olma olasılığı %60'tan yüksekse uygun
    let reason: string | undefined = undefined;
    if (!isAppropriate) {
      const keywords = ["nefret", "şiddet", "küfür", "ırkçılık"];
      if (keywords.some((k) => content.toLowerCase().includes(k))) {
        reason = "Kötü içerik (nefret söylemi veya uygunsuz kelimeler)";
      } else {
        reason = "Genel uygunsuzluk (düşük uygunluk skoru)";
      }
    }

    return { isAppropriate, score: scores[0], reason };
  }

  // Modeli eğit (örnek veriyle)
  async trainModerationModel(trainingData: { text: string; isAppropriate: boolean }[]) {
    if (!this.model || !this.encoder) throw new Error("Model veya encoder başlatılmadı");

    const inputs = [];
    const labels = [];
    for (const data of trainingData) {
      const embeddings = await this.encoder.embed([data.text]);
      const input = embeddings.arraySync()[0] as number[];
      inputs.push(input);

      const labelVector = data.isAppropriate ? [1, 0] : [0, 1]; // [Uygun, Uygunsuz]
      labels.push(labelVector);
    }

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 512]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 2]);

    await this.model.fit(inputTensor, labelTensor, {
      epochs: 150,
      batchSize: 32,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}, accuracy = ${log.acc}`),
      },
    });

    await this.model.save("file://./moderationModel");
    console.log("İçerik moderasyon modeli eğitimi tamamlandı ve kaydedildi.");
  }
}

// Örnek kullanım
const contentModerator = new ContentModerator();
contentModerator.moderateContent("Nefret söylemi içerir, şiddeti teşvik eder!").then(console.log);