import * as tf from "@tensorflow/tfjs";
import * as use from "@tensorflow-models/universal-sentence-encoder";

// Metin çevirisi modeli
export class TextTranslator {
  private model: tf.LayersModel | null = null;
  private encoder: any;
  private vocabulary: { [key: string]: number } = {};
  private reverseVocabulary: { [key: number]: string } = {};

  constructor() {
    this.initializeModel();
    this.initializeVocabulary();
  }

  private async initializeModel() {
    // Universal Sentence Encoder yükle
    this.encoder = await use.load();

    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ units: 512, activation: "relu", inputShape: [512] })); // Giriş vektörü
    this.model.add(tf.layers.dropout({ rate: 0.2 }));
    this.model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    this.model.add(tf.layers.dense({ units: 1000, activation: "softmax" })); // Çıkış kelime sayısına göre
    this.model.compile({ optimizer: "adam", loss: "categoricalCrossentropy", metrics: ["accuracy"] });
  }

  private initializeVocabulary() {
    // Basit bir İngilizce-Türkçe kelime hazinesi (örnek)
    const vocab = {
      "hello": 1, "world": 2, "science": 3, "sport": 4, "politics": 5,
      "merhaba": 6, "dünya": 7, "bilim": 8, "spor": 9, "siyaset": 10,
    };
    this.vocabulary = vocab;
    this.reverseVocabulary = Object.fromEntries(Object.entries(vocab).map(([k, v]) => [v, k]));
  }

  async translateText(content: string, targetLanguage: "tr" | "en" = "tr"): Promise<string> {
    if (!this.model || !this.encoder) throw new Error("Model veya encoder başlatılmadı");

    // Metni vektörleştir
    const embeddings = await this.encoder.embed([content]);
    const inputVector = embeddings.arraySync()[0] as number[];

    const input = tf.tensor2d([inputVector], [1, 512]);
    const prediction = this.model.predict(input) as tf.Tensor;
    const scores = prediction.dataSync() as Float32Array;

    // En yüksek olasılıklı kelimeleri seç
    const translatedTokens: number[] = [];
    for (let i = 0; i < Math.min(10, scores.length); i++) { // Maksimum 10 kelime
      const maxIndex = scores.indexOf(Math.max(...scores));
      translatedTokens.push(maxIndex);
      scores[maxIndex] = -Infinity; // Tekrarı engelle
    }

    // Kelimeleri birleştir
    const translatedWords = translatedTokens.map((token) => this.reverseVocabulary[token] || "unknown");
    return translatedWords.join(" ").replace(/unknown/g, "").trim();
  }

  // Modeli eğit (örnek veriyle)
  async trainTranslatorModel(trainingData: { source: string; target: string }[]) {
    if (!this.model || !this.encoder) throw new Error("Model veya encoder başlatılmadı");

    const inputs = [];
    const labels = [];
    for (const data of trainingData) {
      const sourceEmbeddings = await this.encoder.embed([data.source]);
      const input = sourceEmbeddings.arraySync()[0] as number[];
      inputs.push(input);

      const targetTokens = data.target.split(" ").map((word) => this.vocabulary[word] || 0);
      const labelVector = new Array(1000).fill(0); // Çıkış boyutu
      targetTokens.forEach((token, i) => {
        if (i < 10) labelVector[token] = 1; // İlk 10 kelime için
      });
      labels.push(labelVector);
    }

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 512]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1000]);

    await this.model.fit(inputTensor, labelTensor, {
      epochs: 200,
      batchSize: 32,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}, accuracy = ${log.acc}`),
      },
    });

    await this.model.save("file://./translatorModel");
    console.log("Metin çeviri modeli eğitimi tamamlandı ve kaydedildi.");
  }
}

// Örnek kullanım
const textTranslator = new TextTranslator();
textTranslator.translateText("Hello world, science is great!", "tr").then(console.log);