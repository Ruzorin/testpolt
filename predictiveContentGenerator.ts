import * as tf from "@tensorflow/tfjs";
import * as use from "@tensorflow-models/universal-sentence-encoder";
import { UserBehavior } from "./behaviorPredictor";
import { Post } from "../server/models/Post";

// Tahminleyici içerik üretim modeli
export class PredictiveContentGenerator {
  private model: tf.LayersModel | null = null;
  private encoder: any;
  private vocabulary: { [key: string]: number } = {};
  private reverseVocabulary: { [key: number]: string } = {};

  constructor() {
    this.initializeModel();
    this.initializeVocabulary();
  }

  private async initializeModel() {
    this.encoder = await use.load();

    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ units: 512, activation: "relu", inputShape: [512] })); // Kullanıcı + içerik vektörü
    this.model.add(tf.layers.dropout({ rate: 0.2 }));
    this.model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    this.model.add(tf.layers.dense({ units: 1000, activation: "softmax" })); // Çıkış kelime sayısına göre
    this.model.compile({ optimizer: "adam", loss: "categoricalCrossentropy", metrics: ["accuracy"] });
  }

  private initializeVocabulary() {
    // Basit bir kelime hazinesi (örnek)
    const vocab = {
      "bilim": 1, "spor": 2, "siyaset": 3, "eğlence": 4, "futbol": 5,
      "keşif": 6, "maç": 7, "haber": 8, "oyun": 9, "video": 10,
    };
    this.vocabulary = vocab;
    this.reverseVocabulary = Object.fromEntries(Object.entries(vocab).map(([k, v]) => [v, k]));
  }

  async generateContent(user: UserBehavior, maxLength: number = 50): Promise<string> {
    if (!this.model || !this.encoder) throw new Error("Model veya encoder başlatılmadı");

    // Kullanıcı ve mevcut içerik vektörünü birleştir
    const userVector = this.vectorizeUser(user);
    const contentVector = new Array(512).fill(0); // Başlangıç vektörü

    const combinedVector = [...contentVector, ...userVector];
    const input = tf.tensor2d([combinedVector], [1, 512 + 15]);
    const prediction = this.model.predict(input) as tf.Tensor;
    const scores = prediction.dataSync() as Float32Array;

    // En yüksek olasılıklı kelimeleri seç
    const generatedTokens: number[] = [];
    for (let i = 0; i < Math.min(maxLength / 5, 1000); i++) { // Maksimum 10 kelime
      const maxIndex = scores.indexOf(Math.max(...scores));
      generatedTokens.push(maxIndex);
      scores[maxIndex] = -Infinity; // Tekrarı engelle
    }

    // Kelimeleri birleştir
    const generatedWords = generatedTokens.map((token) => this.reverseVocabulary[token] || "unknown");
    return generatedWords.join(" ").replace(/unknown/g, "").trim().substring(0, maxLength);
  }

  private vectorizeUser(user: UserBehavior): number[] {
    const vector = new Array(15).fill(0);
    const categories = ["Bilim", "Spor", "Siyaset", "Eğlence", "Futbol"];

    // Kategorilere göre ağırlık
    user.selectedCategories.forEach((cat, i) => {
      if (categories.includes(cat)) vector[i] = 1;
    });

    // Etkileşim sayıları
    vector[5] = user.likedPosts.length / 100; // Normalize
    vector[6] = user.viewedPosts.length / 100;
    vector[7] = user.interactions.filter((i) => i.action === "share").length / 50;

    // Zaman tabanlı özellikler
    const now = Date.now();
    vector[8] = (now - user.lastLogin) / (24 * 60 * 60 * 1000); // Gün cinsinden
    vector[9] = user.sessionDuration / 3600; // Saat cinsinden

    // Son etkileşimlerin zaman analizi
    const recentInteractions = user.interactions
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(0, 5)
      .map((i) => (now - i.timestamp) / (60 * 60 * 1000)); // Saat cinsinden
    recentInteractions.forEach((time, i) => {
      vector[10 + i] = time / 24; // Gün cinsinden normalize
    });

    return vector;
  }

  // Modeli eğit (örnek veriyle)
  async trainContentGeneratorModel(trainingData: { user: UserBehavior; content: string }[]) {
    if (!this.model || !this.encoder) throw new Error("Model veya encoder başlatılmadı");

    const inputs = [];
    const labels = [];
    for (const data of trainingData) {
      const userVector = this.vectorizeUser(data.user);
      const contentEmbeddings = await this.encoder.embed([data.content]);
      const contentVector = contentEmbeddings.arraySync()[0] as number[];
      const combinedVector = [...contentVector, ...userVector];
      inputs.push(combinedVector);

      const tokens = data.content.split(" ").map((word) => this.vocabulary[word] || 0);
      const labelVector = new Array(1000).fill(0); // Çıkış boyutu
      tokens.forEach((token, i) => {
        if (i < 10) labelVector[token] = 1; // İlk 10 kelime için
      });
      labels.push(labelVector);
    }

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 527]); // 512 + 15
    const labelTensor = tf.tensor2d(labels, [labels.length, 1000]);

    await this.model.fit(inputTensor, labelTensor, {
      epochs: 200,
      batchSize: 32,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}, accuracy = ${log.acc}`),
      },
    });

    await this.model.save("file://./contentGeneratorModel");
    console.log("Tahminleyici içerik üretim modeli eğitimi tamamlandı ve kaydedildi.");
  }
}

// Örnek kullanım
const predictiveContentGenerator = new PredictiveContentGenerator();
const userBehavior: UserBehavior = {
  address: "0x789",
  viewedPosts: ["1"],
  likedPosts: ["2"],
  selectedCategories: ["Bilim", "Futbol"],
  interactions: [
    { postId: "1", timestamp: Date.now() - 1000 * 60 * 60, action: "view" },
    { postId: "2", timestamp: Date.now() - 2000 * 60 * 60, action: "like" },
  ],
  lastLogin: Date.now(),
  sessionDuration: 1800,
};
predictiveContentGenerator.generateContent(userBehavior, 50).then(console.log);