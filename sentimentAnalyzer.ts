import * as tf from "@tensorflow/tfjs";
import * as use from "@tensorflow-models/universal-sentence-encoder";

// Duygu analizi modeli
export class SentimentAnalyzer {
  private model: tf.LayersModel | null = null;
  private encoder: any;

  constructor() {
    this.initializeModel();
  }

  private async initializeModel() {
    // Universal Sentence Encoder yükle
    this.encoder = await use.load();

    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ units: 256, activation: "relu", inputShape: [512] })); // USE çıkış boyutu
    this.model.add(tf.layers.dropout({ rate: 0.2 }));
    this.model.add(tf.layers.dense({ units: 128, activation: "relu" }));
    this.model.add(tf.layers.dense({ units: 3, activation: "softmax" })); // Olumlu, Nötr, Olumsuz
    this.model.compile({ optimizer: "adam", loss: "categoricalCrossentropy", metrics: ["accuracy"] });
  }

  async analyzeSentiment(content: string): Promise<{ sentiment: "positive" | "neutral" | "negative"; score: number }> {
    if (!this.model || !this.encoder) throw new Error("Model veya encoder başlatılmadı");

    // Metni vektörleştir
    const embeddings = await this.encoder.embed([content]);
    const input = embeddings.arraySync()[0] as number[];

    const inputTensor = tf.tensor2d([input], [1, 512]);
    const prediction = this.model.predict(inputTensor) as tf.Tensor;
    const scores = prediction.dataSync() as Float32Array;

    const sentiments = ["negative", "neutral", "positive"];
    const maxIndex = scores.indexOf(Math.max(...scores));
    return { sentiment: sentiments[maxIndex] as "positive" | "neutral" | "negative", score: scores[maxIndex] };
  }

  // Modeli eğit (örnek veriyle)
  async trainSentimentModel(trainingData: { text: string; sentiment: "positive" | "neutral" | "negative" }[]) {
    if (!this.model || !this.encoder) throw new Error("Model veya encoder başlatılmadı");

    const inputs = [];
    const labels = [];
    for (const data of trainingData) {
      const embeddings = await this.encoder.embed([data.text]);
      const input = embeddings.arraySync()[0] as number[];
      inputs.push(input);

      const labelVector = [0, 0, 0];
      const sentimentIndex = ["negative", "neutral", "positive"].indexOf(data.sentiment);
      if (sentimentIndex !== -1) labelVector[sentimentIndex] = 1;
      labels.push(labelVector);
    }

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 512]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 3]);

    await this.model.fit(inputTensor, labelTensor, {
      epochs: 150,
      batchSize: 32,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}, accuracy = ${log.acc}`),
      },
    });

    await this.model.save("file://./sentimentModel");
    console.log("Duygu analizi modeli eğitimi tamamlandı ve kaydedildi.");
  }
}

// Örnek kullanım
const sentimentAnalyzer = new SentimentAnalyzer();
sentimentAnalyzer.analyzeSentiment("Harika bir gün, çok mutluyum!").then(console.log);