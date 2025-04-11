import * as tf from "@tensorflow/tfjs";
import { UserBehavior } from "./behaviorPredictor";
import { Market } from "../server/models/Market";

// Tahminleyici analitik modeli
export class PredictiveAnalytics {
  private model: tf.LayersModel | null = null;

  constructor() {
    this.initializeModel();
  }

  private initializeModel() {
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ units: 128, activation: "relu", inputShape: [20] })); // Kullanıcı + piyasa özellikleri
    this.model.add(tf.layers.dropout({ rate: 0.3 }));
    this.model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    this.model.add(tf.layers.dense({ units: 2, activation: "softmax" })); // Yükseliş, Düşüş
    this.model.compile({ optimizer: "adam", loss: "categoricalCrossentropy", metrics: ["accuracy"] });
  }

  async predictMarketTrend(user: UserBehavior, market: Market): Promise<{ trend: "rising" | "declining"; confidence: number }> {
    if (!this.model) throw new Error("Model başlatılmadı");

    const userVector = this.vectorizeUser(user);
    const marketVector = this.vectorizeMarket(market);
    const combinedVector = [...userVector, ...marketVector];

    const input = tf.tensor2d([combinedVector], [1, 20]);
    const prediction = this.model.predict(input) as tf.Tensor;
    const scores = prediction.dataSync() as Float32Array;

    const trends = ["declining", "rising"];
    const maxIndex = scores.indexOf(Math.max(...scores));
    return { trend: trends[maxIndex] as "rising" | "declining", confidence: scores[maxIndex] * 100 };
  }

  private vectorizeUser(user: UserBehavior): number[] {
    const vector = new Array(10).fill(0);
    const categories = ["Bilim", "Spor", "Siyaset", "Eğlence", "Futbol"];

    // Kategorilere göre ağırlık
    user.selectedCategories.forEach((cat, i) => {
      if (categories.includes(cat)) vector[i] = 1;
    });

    // Etkileşim sayıları
    vector[5] = user.likedPosts.length / 100; // Normalize
    vector[6] = user.viewedPosts.length / 100;

    // Zaman tabanlı özellikler
    const now = Date.now();
    vector[7] = (now - user.lastLogin) / (24 * 60 * 60 * 1000); // Gün cinsinden
    vector[8] = user.sessionDuration / 3600; // Saat cinsinden
    vector[9] = user.interactions.length / 50; // Normalize

    return vector;
  }

  private vectorizeMarket(market: Market): number[] {
    const vector = new Array(10).fill(0);

    // Piyasa özellikleri
    vector[0] = market.orders.length / 1000; // Sipariş sayısı normalize
    vector[1] = market.resolved ? 1 : 0; // Çözüldü mü?
    vector[2] = market.outcome / 2; // Sonuç (0-1 normalize)
    vector[3] = new Date(market.createdAt).getTime() / (24 * 60 * 60 * 1000); // Gün cinsinden

    // Ortalama fiyat ve hacim (örnek)
    const totalPrice = market.orders.reduce((sum, order) => sum + order.price * order.amount, 0);
    const totalAmount = market.orders.reduce((sum, order) => sum + order.amount, 0);
    vector[4] = totalPrice / 1000; // Ortalama fiyat normalize
    vector[5] = totalAmount / 1000; // Toplam hacim normalize

    // Bid/Ask oranları
    const bids = market.orders.filter((o) => o.isBid);
    const asks = market.orders.filter((o) => !o.isBid);
    vector[6] = bids.length / (bids.length + asks.length) || 0; // Bid oranı
    vector[7] = asks.length / (bids.length + asks.length) || 0; // Ask oranı

    // Son sipariş zamanı
    const latestOrder = market.orders.reduce((max, order) => Math.max(max, new Date(order.createdAt || market.createdAt).getTime()), 0);
    vector[8] = (Date.now() - latestOrder) / (24 * 60 * 60 * 1000); // Gün cinsinden
    vector[9] = market.orders.length > 0 ? 1 : 0; // Aktiflik

    return vector;
  }

  // Modeli eğit (örnek veriyle)
  async trainPredictiveModel(trainingData: { user: UserBehavior; market: Market; trend: "rising" | "declining" }[]) {
    if (!this.model) throw new Error("Model başlatılmadı");

    const inputs = [];
    const labels = [];
    for (const data of trainingData) {
      const userVector = this.vectorizeUser(data.user);
      const marketVector = this.vectorizeMarket(data.market);
      const combinedVector = [...userVector, ...marketVector];
      inputs.push(combinedVector);

      const labelVector = ["declining", "rising"].indexOf(data.trend) === 0 ? [1, 0] : [0, 1];
      labels.push(labelVector);
    }

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 20]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 2]);

    await this.model.fit(inputTensor, labelTensor, {
      epochs: 200,
      batchSize: 32,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}, accuracy = ${log.acc}`),
      },
    });

    await this.model.save("file://./predictiveAnalyticsModel");
    console.log("Tahminleyici analitik modeli eğitimi tamamlandı ve kaydedildi.");
  }
}

// Örnek kullanım
const predictiveAnalytics = new PredictiveAnalytics();
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
const market: Market = {
  id: "market_1",
  question: "Bilimsel keşif başarılı olacak mı?",
  resolved: false,
  outcome: 0,
  orders: [{ user: "0x123", outcome: 1, isBid: true, price: 0.6, amount: 100, createdAt: new Date().toISOString() }],
  createdAt: new Date().toISOString(),
};
predictiveAnalytics.predictMarketTrend(userBehavior, market).then(console.log);