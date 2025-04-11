import * as tf from "@tensorflow/tfjs";
import { Post } from "../server/models/Post";

// Kullanıcı davranış verisi
export interface UserBehavior {
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

export class RecommendationEngine {
  private model: tf.LayersModel | null = null;

  constructor() {
    this.initializeModel();
  }

  private async initializeModel() {
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ units: 64, activation: "relu", inputShape: [10] })); // Kullanıcı özellikleri
    this.model.add(tf.layers.dense({ units: 32, activation: "relu" }));
    this.model.add(tf.layers.dense({ units: posts.length, activation: "softmax" })); // Post sayısı kadar çıkış
    this.model.compile({ optimizer: "adam", loss: "categoricalCrossentropy", metrics: ["accuracy"] });
  }

  async getRecommendations(user: UserBehavior): Promise<Post[]> {
    if (!this.model) throw new Error("Model başlatılmadı");

    // Kullanıcı özelliklerini vektörleştir
    const userVector = this.vectorizeUser(user);
    const input = tf.tensor2d([userVector], [1, 10]);
    const prediction = this.model.predict(input) as tf.Tensor;
    const scores = prediction.dataSync() as Float32Array;

    const recommendations = [];
    for (let i = 0; i < scores.length; i++) {
      if (scores[i] > 0.1) recommendations.push(posts[i]); // Eşik değeri 0.1
    }

    return recommendations.sort((a, b) => {
      const aScore = scores[posts.indexOf(a)];
      const bScore = scores[posts.indexOf(b)];
      return bScore - aScore;
    }).slice(0, 10);
  }

  private vectorizeUser(user: UserBehavior): number[] {
    const vector = new Array(10).fill(0);
    const categories = ["Bilim", "Spor", "Siyaset", "Eğlence", "Futbol"];

    // Kategorilere göre ağırlık
    user.selectedCategories.forEach((cat, i) => {
      if (categories.includes(cat)) vector[i] = 1;
    });

    // Beğeni ve görüntüleme sayıları
    vector[5] = user.likedPosts.length / 100; // Normalize
    vector[6] = user.viewedPosts.length / 100;

    // Son etkileşim zamanına göre ağırlık
    const latestInteraction = user.interactions.reduce((max, curr) => Math.max(max, curr.timestamp), 0);
    vector[7] = latestInteraction ? (Date.now() - latestInteraction) / (24 * 60 * 60 * 1000) : 0; // Gün cinsinden

    return vector;
  }

  // Modeli eğit (örnek veriyle)
  async trainRecommendationModel(trainingData: { user: UserBehavior; postIds: string[] }[]) {
    if (!this.model) throw new Error("Model başlatılmadı");

    const inputs = [];
    const labels = [];
    for (const data of trainingData) {
      const userVector = this.vectorizeUser(data.user);
      const labelVector = new Array(posts.length).fill(0);
      data.postIds.forEach((id) => {
        const index = posts.findIndex((p) => p.id === id);
        if (index !== -1) labelVector[index] = 1;
      });
      inputs.push(userVector);
      labels.push(labelVector);
    }

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 10]);
    const labelTensor = tf.tensor2d(labels, [labels.length, posts.length]);

    await this.model.fit(inputTensor, labelTensor, {
      epochs: 100,
      batchSize: 32,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}, accuracy = ${log.acc}`),
      },
    });

    await this.model.save("file://./recommendationModel");
    console.log("Öneri modeli eğitimi tamamlandı ve kaydedildi.");
  }
}

// Örnek kullanım
const recommendationEngine = new RecommendationEngine();
const userBehavior: UserBehavior = {
  address: "0x789",
  viewedPosts: ["1"],
  likedPosts: ["2"],
  selectedCategories: ["Bilim", "Futbol"],
  interactions: [{ postId: "1", timestamp: Date.now() - 1000 * 60 * 60, action: "view" }],
};
recommendationEngine.getRecommendations(userBehavior).then(console.log);