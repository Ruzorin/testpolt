import * as tf from "@tensorflow/tfjs";

// Spam filtreleme modeli
export class SpamFilter {
  private model: tf.LayersModel | null = null;

  constructor() {
    this.initializeModel();
  }

  private async initializeModel() {
    this.model = tf.sequential();
    this.model.add(tf.layers.embedding({ inputDim: 10000, outputDim: 64, inputLength: 100 })); // Kelime vektörleri
    this.model.add(tf.layers.lstm({ units: 128, returnSequences: true }));
    this.model.add(tf.layers.lstm({ units: 64 }));
    this.model.add(tf.layers.dense({ units: 32, activation: "relu" }));
    this.model.add(tf.layers.dense({ units: 2, activation: "softmax" })); // Spam (1) veya Değil (0)
    this.model.compile({ optimizer: "adam", loss: "categoricalCrossentropy", metrics: ["accuracy"] });
  }

  async isSpam(content: string): Promise<boolean> {
    if (!this.model) throw new Error("Model başlatılmadı");

    // Metni tokenize et ve vektörleştir
    const tokens = content.toLowerCase().split(" ").slice(0, 100); // İlk 100 kelime
    const vocabulary = new Set<string>(["spam", "reklam", "satış", "ücretsiz", "hızlı", "para", "kazan", "dolandırıcı"]); // Örnek kelime seti
    const vector = tokens.map((token) => (vocabulary.has(token) ? 1 : 0));

    const input = tf.tensor2d([vector], [1, 100]);
    const prediction = this.model.predict(input) as tf.Tensor;
    const scores = prediction.dataSync() as Float32Array;

    return scores[1] > 0.7; // Spam olma olasılığı %70'ten yüksekse spam olarak değerlendir
  }

  // Modeli eğit (örnek veriyle)
  async trainSpamModel(trainingData: { text: string; isSpam: boolean }[]) {
    if (!this.model) throw new Error("Model başlatılmadı");

    const inputs = [];
    const labels = [];
    for (const data of trainingData) {
      const tokens = data.text.toLowerCase().split(" ").slice(0, 100);
      const vector = tokens.map((token) => (vocabulary.has(token) ? 1 : 0));
      inputs.push(vector);
      labels.push(data.isSpam ? [0, 1] : [1, 0]); // [Değil, Spam]
    }

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 100]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 2]);

    await this.model.fit(inputTensor, labelTensor, {
      epochs: 100,
      batchSize: 32,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}, accuracy = ${log.acc}`),
      },
    });

    await this.model.save("file://./spamModel");
    console.log("Spam filtreleme modeli eğitimi tamamlandı ve kaydedildi.");
  }
}

// Örnek kullanım
const spamFilter = new SpamFilter();
spamFilter.isSpam("Ücretsiz para kazan, hızlı zengin ol!").then(console.log);