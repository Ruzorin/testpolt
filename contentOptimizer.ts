import * as tf from "@tensorflow/tfjs";
import * as use from "@tensorflow-models/universal-sentence-encoder";
import { Post } from "../server/models/Post";

// İçerik optimizasyon modeli
export class ContentOptimizer {
  private model: tf.LayersModel | null = null;
  private encoder: any;

  constructor() {
    this.initializeModel();
  }

  private async initializeModel() {
    // Universal Sentence Encoder yükle
    this.encoder = await use.load();

    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ units: 256, activation: "relu", inputShape: [512] }));
    this.model.add(tf.layers.dropout({ rate: 0.2 }));
    this.model.add(tf.layers.dense({ units: 128, activation: "relu" }));
    this.model.add(tf.layers.dense({ units: 3, activation: "softmax" })); // Yüksek, Orta, Düşük performans
    this.model.compile({ optimizer: "adam", loss: "categoricalCrossentropy", metrics: ["accuracy"] });
  }

  async optimizeContent(content: string, userBehavior: Partial<UserBehavior>): Promise<{
    performance: "high" | "medium" | "low";
    suggestions: string[];
  }> {
    if (!this.model || !this.encoder) throw new Error("Model veya encoder başlatılmadı");

    // Metni vektörleştir ve kullanıcı davranışını birleştir
    const embeddings = await this.encoder.embed([content]);
    const contentVector = embeddings.arraySync()[0] as number[];

    const behaviorVector = this.vectorizeBehavior(userBehavior);
    const combinedVector = [...contentVector, ...behaviorVector];

    const input = tf.tensor2d([combinedVector], [1, 512 + 5]); // 512 (USE) + 5 (davranış)
    const prediction = this.model.predict(input) as tf.Tensor;
    const scores = prediction.dataSync() as Float32Array;

    const performanceLevels = ["low", "medium", "high"];
    const maxIndex = scores.indexOf(Math.max(...scores));
    const performance = performanceLevels[maxIndex] as "high" | "medium" | "low";

    const suggestions = this.generateSuggestions(content, performance, userBehavior);
    return { performance, suggestions };
  }

  private vectorizeBehavior(userBehavior: Partial<UserBehavior>): number[] {
    const vector = new Array(5).fill(0);
    const categories = ["Bilim", "Spor", "Siyaset", "Eğlence", "Futbol"];

    if (userBehavior.selectedCategories) {
      userBehavior.selectedCategories.forEach((cat, i) => {
        if (categories.includes(cat)) vector[i] = 1;
      });
    }

    if (userBehavior.likedPosts) vector[4] = userBehavior.likedPosts.length / 100; // Normalize
    return vector;
  }

  private generateSuggestions(content: string, performance: "high" | "medium" | "low", userBehavior: Partial<UserBehavior>): string[] {
    const suggestions: string[] = [];
    if (performance === "low") {
      suggestions.push("Daha ilgi çekici anahtar kelimeler ekleyin.");
      if (userBehavior.selectedCategories) {
        suggestions.push(`Kullanıcının ilgilendiği kategoriler (${userBehavior.selectedCategories.join(", ")}) için içeriği optimize edin.`);
      }
    } else if (performance === "medium") {
      suggestions.push("İçeriği görseller veya videolarla zenginleştirin.");
      suggestions.push("Daha fazla etkileşim için çağrı eylemleri ekleyin.");
    }
    return suggestions;
  }

  // Modeli eğit (örnek veriyle)
  async trainContentOptimizer(trainingData: { content: string; userBehavior: Partial<UserBehavior>; performance: "high" | "medium" | "low" }[]) {
    if (!this.model || !this.encoder) throw new Error("Model veya encoder başlatılmadı");

    const inputs = [];
    const labels = [];
    for (const data of trainingData) {
      const embeddings = await this.encoder.embed([data.content]);
      const contentVector = embeddings.arraySync()[0] as number[];
      const behaviorVector = this.vectorizeBehavior(data.userBehavior);
      const combinedVector = [...contentVector, ...behaviorVector];
      inputs.push(combinedVector);

      const labelVector = [0, 0, 0];
      const performanceIndex = ["low", "medium", "high"].indexOf(data.performance);
      if (performanceIndex !== -1) labelVector[performanceIndex] = 1;
      labels.push(labelVector);
    }

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 517]); // 512 + 5
    const labelTensor = tf.tensor2d(labels, [labels.length, 3]);

    await this.model.fit(inputTensor, labelTensor, {
      epochs: 200,
      batchSize: 32,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}, accuracy = ${log.acc}`),
      },
    });

    await this.model.save("file://./contentOptimizerModel");
    console.log("İçerik optimizasyon modeli eğitimi tamamlandı ve kaydedildi.");
  }
}

// Örnek kullanım
const contentOptimizer = new ContentOptimizer();
const userBehavior = { selectedCategories: ["Bilim"], likedPosts: ["1"] };
contentOptimizer.optimizeContent("Bilimsel bir keşif yaptık!", userBehavior).then(console.log);