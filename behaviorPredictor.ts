import * as tf from "@tensorflow/tfjs";
import { Post, User } from "../server/models/Post"; // Varsayılan olarak User modeli de eklendi

// Kullanıcı davranış verisi
interface UserBehavior {
  address: string;
  viewedPosts: string[];
  likedPosts: string[];
  selectedCategories: string[];
  interactions: { postId: string; timestamp: number; action: "view" | "like" | "share" }[];
  lastLogin: number;
  sessionDuration: number;
}

// Örnek veri (gerçek sistemde MongoDB'den çekilecek)
const posts: Post[] = [
  { id: "1", type: "video", content: "Bilim Videosu", category: "Bilim", creator: "0x123", likes: 10, timestamp: "2025-02-24" },
  { id: "2", type: "tweet", content: "Maç yorumu", category: "Futbol", creator: "0x456", likes: 5, timestamp: "2025-02-24" },
];

export class BehaviorPredictor {
  private model: tf.LayersModel | null = null;

  constructor() {
    this.initializeModel();
  }

  private async initializeModel() {
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ units: 128, activation: "relu", inputShape: [15] })); // Kullanıcı özellikleri
    this.model.add(tf.layers.dropout({ rate: 0.3 }));
    this.model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    this.model.add(tf.layers.dense({ units: 3, activation: "softmax" })); // Yüksek, Orta, Düşük etkileşim
    this.model.compile({ optimizer: "adam", loss: "categoricalCrossentropy", metrics: ["accuracy"] });
  }

  async predictBehavior(user: UserBehavior): Promise<{ engagementLevel: "high" | "medium" | "low"; score: number }> {
    if (!this.model) throw new Error("Model başlatılmadı");

    // Kullanıcı özelliklerini vektörleştir
    const userVector = this.vectorizeUser(user);
    const input = tf.tensor2d([userVector], [1, 15]);
    const prediction = this.model.predict(input) as tf.Tensor;
    const scores = prediction.dataSync() as Float32Array;

    const engagementLevels = ["low", "medium", "high"];
    const maxIndex = scores.indexOf(Math.max(...scores));
    return { engagementLevel: engagementLevels[maxIndex] as "high" | "medium" | "low", score: scores[maxIndex] };
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
  async trainBehaviorModel(trainingData: { user: UserBehavior; engagement: "high" | "medium" | "low" }[]) {
    if (!this.model) throw new Error("Model başlatılmadı");

    const inputs = [];
    const labels = [];
    for (const data of trainingData) {
      const userVector = this.vectorizeUser(data.user);
      inputs.push(userVector);

      const labelVector = [0, 0, 0];
      const engagementIndex = ["low", "medium", "high"].indexOf(data.engagement);
      if (engagementIndex !== -1) labelVector[engagementIndex] = 1;
      labels.push(labelVector);
    }

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 15]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 3]);

    await this.model.fit(inputTensor, labelTensor, {
      epochs: 200,
      batchSize: 32,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}, accuracy = ${log.acc}`),
      },
    });

    await this.model.save("file://./behaviorModel");
    console.log("Davranış tahmini modeli eğitimi tamamlandı ve kaydedildi.");
  }
}

// Örnek kullanım
const behaviorPredictor = new BehaviorPredictor();
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
  sessionDuration: 1800, // 30 dakika
};
behaviorPredictor.predictBehavior(userBehavior).then(console.log);