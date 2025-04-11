import * as tf from "@tensorflow/tfjs";
import { UserBehavior } from "./behaviorPredictor";
import { Post } from "../server/models/Post";
import { Server } from "socket.io";

// Kullanıcı etkileşim optimizasyonu
export class UserInteractionOptimizer {
  private model: tf.LayersModel | null = null;
  private io: Server;
  private users: Map<string, UserBehavior> = new Map();

  constructor(io: Server) {
    this.io = io;
    this.initializeModel();
  }

  private initializeModel() {
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ units: 128, activation: "relu", inputShape: [15] })); // Kullanıcı özellikleri
    this.model.add(tf.layers.dropout({ rate: 0.3 }));
    this.model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    this.model.add(tf.layers.dense({ units: 3, activation: "softmax" })); // Etkileşim önerileri: Beğen, Paylaş, İzle
    this.model.compile({ optimizer: "adam", loss: "categoricalCrossentropy", metrics: ["accuracy"] });
  }

  async optimizeInteractions(user: UserBehavior, post: Post): Promise<{ action: "like" | "share" | "view"; score: number }[]> {
    if (!this.model) throw new Error("Model başlatılmadı");

    const userVector = this.vectorizeUser(user);
    const postVector = this.vectorizePost(post);
    const combinedVector = [...userVector, ...postVector];

    const input = tf.tensor2d([combinedVector], [1, 25]); // 15 + 10
    const prediction = this.model.predict(input) as tf.Tensor;
    const scores = prediction.dataSync() as Float32Array;

    const actions = ["view", "like", "share"];
    const recommendations = actions.map((action, i) => ({
      action: action as "like" | "share" | "view",
      score: scores[i] * 100,
    }));

    return recommendations.sort((a, b) => b.score - a.score).slice(0, 3);
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

  private vectorizePost(post: Post): number[] {
    const vector = new Array(10).fill(0);
    const categories = ["Bilim", "Spor", "Siyaset", "Eğlence", "Futbol"];

    // Kategori ağırlığı
    const catIndex = categories.indexOf(post.category);
    if (catIndex !== -1) vector[catIndex] = 1;

    // Etkileşim istatistikleri
    vector[5] = post.likes / 100; // Normalize
    vector[6] = new Date(post.timestamp).getTime() / (24 * 60 * 60 * 1000); // Gün cinsinden

    // İçerik türü (video, tweet, photo)
    if (post.type === "video") vector[7] = 1;
    else if (post.type === "tweet") vector[8] = 1;
    else if (post.type === "photo") vector[9] = 1;

    return vector;
  }

  initializeListeners() {
    this.io.on("connection", (socket) => {
      console.log("Kullanıcı bağlandı (etkileşim optimizasyonu):", socket.id);

      socket.on("userBehavior", async (behavior: Partial<UserBehavior>) => {
        const userAddress = behavior.address || socket.id;
        let fullBehavior = this.users.get(userAddress) || { address: userAddress, viewedPosts: [], likedPosts: [], selectedCategories: [], interactions: [], lastLogin: Date.now(), sessionDuration: 0 };
        fullBehavior = { ...fullBehavior, ...behavior, lastLogin: Date.now(), sessionDuration: (fullBehavior.sessionDuration || 0) + 1 };

        this.users.set(userAddress, fullBehavior);
        this.optimizeInteractionsForUser(userAddress, fullBehavior);
      });

      socket.on("postInteraction", async (data: { postId: string; post: Partial<Post> }) => {
        const post = posts.find((p) => p.id === data.postId) || (data.post as Post);
        for (const [userAddress, behavior] of this.users) {
          const recommendations = await this.optimizeInteractions(behavior, post);
          this.io.to(userAddress).emit("interactionRecommendations", { postId: data.postId, recommendations });
        }
      });

      socket.on("disconnect", () => {
        this.users.delete(userAddress);
        console.log("Kullanıcı ayrıldı (etkileşim optimizasyonu):", socket.id);
      });
    });
  }

  private async optimizeInteractionsForUser(userAddress: string, behavior: UserBehavior) {
    for (const post of posts) {
      const recommendations = await this.optimizeInteractions(behavior, post);
      this.io.to(userAddress).emit("interactionRecommendations", { postId: post.id, recommendations });
    }
  }

  // Modeli güncelle ve eğit (gerçek zamanlı veriyle)
  async updateAndTrainModel(newData: { user: UserBehavior; post: Post; action: "like" | "share" | "view" }[]) {
    if (!this.model) throw new Error("Model başlatılmadı");

    const inputs = [];
    const labels = [];
    for (const data of newData) {
      const userVector = this.vectorizeUser(data.user);
      const postVector = this.vectorizePost(data.post);
      const combinedVector = [...userVector, ...postVector];
      inputs.push(combinedVector);

      const labelVector = [0, 0, 0];
      const actionIndex = ["view", "like", "share"].indexOf(data.action);
      if (actionIndex !== -1) labelVector[actionIndex] = 1;
      labels.push(labelVector);
    }

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 25]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 3]);

    await this.model.fit(inputTensor, labelTensor, {
      epochs: 200,
      batchSize: 32,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}, accuracy = ${log.acc}`),
      },
    });

    await this.model.save("file://./interactionOptimizerModel");
    console.log("Kullanıcı etkileşim optimizasyonu modeli eğitimi tamamlandı ve kaydedildi.");
  }
}

// Örnek kullanım (server tarafında)
import { createServer } from "http";
import { Server } from "socket.io";
const httpServer = createServer();
const io = new Server(httpServer);
const userInteractionOptimizer = new UserInteractionOptimizer(io);
userInteractionOptimizer.initializeListeners();
httpServer.listen(3005, () => console.log("User interaction optimizer server çalışıyor: 3005"));