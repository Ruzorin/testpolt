import * as tf from "@tensorflow/tfjs";
import * as use from "@tensorflow-models/universal-sentence-encoder";
import { Server } from "socket.io";
import { Post } from "../server/models/Post";

// Gerçek zamanlı metin analizi
export class RealTimeTextAnalyzer {
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
    this.model.add(tf.layers.dense({ units: 4, activation: "softmax" })); // Trend, Duygu, Spam, Önem
    this.model.compile({ optimizer: "adam", loss: "categoricalCrossentropy", metrics: ["accuracy"] });
  }

  async analyzeTextInRealTime(content: string, postId: string): Promise<{
    trend: "rising" | "stable" | "declining";
    sentiment: "positive" | "neutral" | "negative";
    isSpam: boolean;
    importance: number;
  }> {
    if (!this.model || !this.encoder) throw new Error("Model veya encoder başlatılmadı");

    const embeddings = await this.encoder.embed([content]);
    const input = tf.tensor2d(embeddings.arraySync() as number[][], [1, 512]);
    const prediction = this.model.predict(input) as tf.Tensor;
    const scores = prediction.dataSync() as Float32Array;

    const trends = ["declining", "stable", "rising"];
    const sentiments = ["negative", "neutral", "positive"];
    const [trendIndex, sentimentIndex, spamIndex, importanceIndex] = [0, 1, 2, 3];

    return {
      trend: trends[Math.round(scores[trendIndex] * 2)] as "rising" | "stable" | "declining",
      sentiment: sentiments[Math.round(scores[sentimentIndex] * 2)] as "positive" | "neutral" | "negative",
      isSpam: scores[spamIndex] > 0.7,
      importance: scores[importanceIndex] * 100, // 0-100 arası önem skoru
    };
  }

  initializeListeners() {
    this.io.on("connection", (socket) => {
      console.log("Kullanıcı bağlandı (real-time metin):", socket.id);

      socket.on("newPost", async (post: Partial<Post>) => {
        const analysis = await this.analyzeTextInRealTime(post.content || "", post.id || "");
        socket.emit("textAnalysis", { postId: post.id, ...analysis });
      });

      socket.on("disconnect", () => {
        console.log("Kullanıcı ayrıldı (real-time metin):", socket.id);
      });
    });
  }

  // Modeli eğit (örnek veriyle)
  async trainTextAnalyzerModel(trainingData: { text: string; trend: "rising" | "stable" | "declining"; sentiment: "positive" | "neutral" | "negative"; isSpam: boolean; importance: number }[]) {
    if (!this.model || !this.encoder) throw new Error("Model veya encoder başlatılmadı");

    const inputs = [];
    const labels = [];
    for (const data of trainingData) {
      const embeddings = await this.encoder.embed([data.text]);
      const input = embeddings.arraySync()[0] as number[];
      inputs.push(input);

      const labelVector = [
        ["declining", "stable", "rising"].indexOf(data.trend) / 2,
        ["negative", "neutral", "positive"].indexOf(data.sentiment) / 2,
        data.isSpam ? 1 : 0,
        data.importance / 100,
      ];
      labels.push(labelVector);
    }

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 512]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 4]);

    await this.model.fit(inputTensor, labelTensor, {
      epochs: 200,
      batchSize: 32,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}, accuracy = ${log.acc}`),
      },
    });

    await this.model.save("file://./textAnalyzerModel");
    console.log("Gerçek zamanlı metin analizi modeli eğitimi tamamlandı ve kaydedildi.");
  }
}

// Örnek kullanım (server tarafında)
import { createServer } from "http";
import { Server } from "socket.io";
const httpServer = createServer();
const io = new Server(httpServer);
const realTimeTextAnalyzer = new RealTimeTextAnalyzer(io);
realTimeTextAnalyzer.initializeListeners();
httpServer.listen(3003, () => console.log("Real-time text analyzer server çalışıyor: 3003"));