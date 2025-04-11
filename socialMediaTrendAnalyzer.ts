import * as tf from "@tensorflow/tfjs";
import * as use from "@tensorflow-models/universal-sentence-encoder";
import { Server } from "socket.io";
import { Post } from "../server/models/Post";

// Sosyal medya trend analizi
export class SocialMediaTrendAnalyzer {
  private model: tf.LayersModel | null = null;
  private encoder: any;
  private io: Server;

  constructor(io: Server) {
    this.io = io;
    this.initializeModel();
  }

  private async initializeModel() {
    this.encoder = await use.load();

    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ units: 256, activation: "relu", inputShape: [512] }));
    this.model.add(tf.layers.dropout({ rate: 0.2 }));
    this.model.add(tf.layers.dense({ units: 128, activation: "relu" }));
    this.model.add(tf.layers.dense({ units: 3, activation: "softmax" })); // Yükselen, Sabit, Düşen
    this.model.compile({ optimizer: "adam", loss: "categoricalCrossentropy", metrics: ["accuracy"] });
  }

  async analyzeTrend(content: string, post: Partial<Post>): Promise<{
    trend: "rising" | "stable" | "declining";
    popularityScore: number;
    relatedTopics: string[];
  }> {
    if (!this.model || !this.encoder) throw new Error("Model veya encoder başlatılmadı");

    const embeddings = await this.encoder.embed([content]);
    const input = tf.tensor2d(embeddings.arraySync() as number[][], [1, 512]);
    const prediction = this.model.predict(input) as tf.Tensor;
    const scores = prediction.dataSync() as Float32Array;

    const trends = ["declining", "stable", "rising"];
    const trendIndex = scores.indexOf(Math.max(...scores));
    const trend = trends[trendIndex] as "rising" | "stable" | "declining";
    const popularityScore = scores[trendIndex] * 100;

    // İlgili konular (örnek basit mantık)
    const topics = ["Bilim", "Spor", "Siyaset", "Eğlence", "Futbol"];
    const relatedTopics = topics.filter((topic) => content.toLowerCase().includes(topic.toLowerCase())).slice(0, 3);

    return { trend, popularityScore, relatedTopics };
  }

  initializeListeners() {
    this.io.on("connection", (socket) => {
      console.log("Kullanıcı bağlandı (trend analizi):", socket.id);

      socket.on("newPost", async (post: Partial<Post>) => {
        if (post.content) {
          const analysis = await this.analyzeTrend(post.content, post);
          socket.emit("trendAnalysis", { postId: post.id, ...analysis });
        }
      });

      socket.on("disconnect", () => {
        console.log("Kullanıcı ayrıldı (trend analizi):", socket.id);
      });
    });
  }

  // Modeli eğit (örnek veriyle)
  async trainTrendAnalyzerModel(trainingData: { text: string; trend: "rising" | "stable" | "declining"; popularity: number }[]) {
    if (!this.model || !this.encoder) throw new Error("Model veya encoder başlatılmadı");

    const inputs = [];
    const labels = [];
    for (const data of trainingData) {
      const embeddings = await this.encoder.embed([data.text]);
      const input = embeddings.arraySync()[0] as number[];
      inputs.push(input);

      const labelVector = [
        ["declining", "stable", "rising"].indexOf(data.trend) / 2,
        data.popularity / 100,
        0, // Ekstra yer tutucu (model genişletilebilir)
      ];
      labels.push(labelVector);
    }

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 512]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 3]);

    await this.model.fit(inputTensor, labelTensor, {
      epochs: 200,
      batchSize: 32,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}, accuracy = ${log.acc}`),
      },
    });

    await this.model.save("file://./trendAnalyzerModel");
    console.log("Sosyal medya trend analizi modeli eğitimi tamamlandı ve kaydedildi.");
  }
}

// Örnek kullanım (server tarafında)
import { createServer } from "http";
import { Server } from "socket.io";
const httpServer = createServer();
const io = new Server(httpServer);
const socialMediaTrendAnalyzer = new SocialMediaTrendAnalyzer(io);
socialMediaTrendAnalyzer.initializeListeners();
httpServer.listen(3004, () => console.log("Social media trend analyzer server çalışıyor: 3004"));