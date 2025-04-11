import * as tf from "@tensorflow/tfjs";
import { UserBehavior } from "./behaviorPredictor";
import { Post } from "../server/models/Post";

// Kullanıcı kişiselleştirme modeli
export class UserPersonalization {
  private model: tf.LayersModel | null = null;

  constructor() {
    this.initializeModel();
  }

  private initializeModel() {
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ units: 128, activation: "relu", inputShape: [15] })); // Kullanıcı özellikleri
    this.model.add(tf.layers.dropout({ rate: 0.3 }));
    this.model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    this.model.add(tf.layers.dense({ units: posts.length, activation: "softmax" })); // Post sayısı kadar çıkış
    this.model.compile({ optimizer: "adam", loss: "categoricalCrossentropy", metrics: ["accuracy"] });
  }

  async personalizeContent(user: UserBehavior): Promise<{ postId: string; score: number }[]> {
    if (!this.model) throw new Error("Model başlatılmadı");

    const userVector = this.vectorizeUser(user);
    const input = tf.tensor2d([userVector], [1, 15]);
    const prediction = this.model.predict(input) as tf.Tensor;
    const scores = prediction.dataSync() as Float32Array;

    const recommendations = [];
    for (let i = 0; i < scores.length; i++) {
      if (scores[i] > 0.1) recommendations.push({ postId: posts[i].id, score: scores[i] * 100 });
    }

    return recommendations.sort((a, b) => b.score - a.score).slice(0, 10);
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
  async trainPersonalizationModel(trainingData: { user: UserBehavior; postIds: string[] }[]) {
    if (!this.model) throw new Error("Model başlatılmadı");

    const inputs = [];
    const labels = [];
    for (const data of trainingData) {
      const userVector = this.vectorizeUser(data.user);
      inputs.push(userVector);

      const labelVector = new Array(posts.length).fill(0);
      data.postIds.forEach((id) => {
        const index = posts.findIndex((p) => p.id === id);
        if (index !== -1) labelVector[index] = 1;
      });
      labels.push(labelVector);
    }

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 15]);
    const labelTensor = tf.tensor2d(labels, [labels.length, posts.length]);

    await this.model.fit(inputTensor, labelTensor, {
      epochs: 200,
      batchSize: 32,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}, accuracy = ${log.acc}`),
      },
    });

    await this.model.save("file://./personalizationModel");
    console.log("Kullanıcı kişiselleştirme modeli eğitimi tamamlandı ve kaydedildi.");
  }
}

// Örnek kullanım
const userPersonalization = new UserPersonalization();
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
userPersonalization.personalizeContent(userBehavior).then(console.log);