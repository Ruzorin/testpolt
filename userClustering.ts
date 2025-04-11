import * as tf from "@tensorflow/tfjs";
import { UserBehavior } from "./behaviorPredictor";

// Kullanıcı kümelenmesi modeli
export class UserClustering {
  private kmeans: tf.KMeans | null = null;
  private numClusters: number = 5; // Kümelerin sayısı

  constructor() {
    this.initializeModel();
  }

  private initializeModel() {
    this.kmeans = new tf.KMeans({ k: this.numClusters });
  }

  async clusterUsers(users: UserBehavior[]): Promise<{ cluster: number; user: UserBehavior }[]> {
    if (!this.kmeans) throw new Error("Model başlatılmadı");

    // Kullanıcıları vektörleştir
    const vectors = users.map((user) => this.vectorizeUser(user));
    const dataTensor = tf.tensor2d(vectors, [vectors.length, 15]); // Davranış vektörü boyutu

    // K-Means ile kümelenme
    const clusters = await this.kmeans.fit(dataTensor, { epochs: 100, batchSize: 32 });
    const assignments = clusters.assignments.dataSync() as Float32Array;

    // Sonuçları birleştir
    const clusteredUsers = users.map((user, i) => ({
      cluster: assignments[i],
      user,
    }));

    dataTensor.dispose();
    return clusteredUsers;
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

  // Kümelenme verisiyle modeli eğit (örnek)
  async trainClusteringModel(trainingData: UserBehavior[]) {
    if (!this.kmeans) throw new Error("Model başlatılmadı");

    const vectors = trainingData.map((user) => this.vectorizeUser(user));
    const dataTensor = tf.tensor2d(vectors, [vectors.length, 15]);

    await this.kmeans.fit(dataTensor, { epochs: 100, batchSize: 32 });
    console.log("Kullanıcı kümelenme modeli eğitimi tamamlandı.");

    dataTensor.dispose();
  }
}

// Örnek kullanım
const userClustering = new UserClustering();
const users: UserBehavior[] = [
  {
    address: "0x123",
    viewedPosts: ["1"],
    likedPosts: ["2"],
    selectedCategories: ["Bilim"],
    interactions: [{ postId: "1", timestamp: Date.now(), action: "view" }],
    lastLogin: Date.now(),
    sessionDuration: 1800,
  },
  {
    address: "0x456",
    viewedPosts: ["2"],
    likedPosts: [],
    selectedCategories: ["Futbol"],
    interactions: [{ postId: "2", timestamp: Date.now(), action: "view" }],
    lastLogin: Date.now(),
    sessionDuration: 1200,
  },
];
userClustering.clusterUsers(users).then(console.log);