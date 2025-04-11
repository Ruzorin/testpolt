import * as tf from "@tensorflow/tfjs";
import * as use from "@tensorflow-models/universal-sentence-encoder";

// Metin özetleme modeli
export class TextSummarizer {
  private model: tf.LayersModel | null = null;
  private encoder: any;

  constructor() {
    this.initializeModel();
  }

  private async initializeModel() {
    // Universal Sentence Encoder yükle
    this.encoder = await use.load();

    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ units: 512, activation: "relu", inputShape: [512] })); // USE çıkış boyutu
    this.model.add(tf.layers.dropout({ rate: 0.2 }));
    this.model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    this.model.add(tf.layers.dense({ units: 128, activation: "relu" }));
    this.model.add(tf.layers.dense({ units: 512, activation: "linear" })); // Çıkış, orijinal boyut
    this.model.compile({ optimizer: "adam", loss: "meanSquaredError", metrics: ["mae"] });
  }

  async summarizeText(content: string, maxLength: number = 50): Promise<string> {
    if (!this.model || !this.encoder) throw new Error("Model veya encoder başlatılmadı");

    // Metni cümlelere böl
    const sentences = content.split(/[.!?]+/).filter((s) => s.trim().length > 0).slice(0, 10); // İlk 10 cümle
    if (sentences.length === 0) return content;

    // Her cümleyi vektörleştir
    const embeddings = await this.encoder.embed(sentences);
    const sentenceVectors = embeddings.arraySync() as number[][];

    // Önem skorlarını hesapla
    const input = tf.tensor2d(sentenceVectors, [sentenceVectors.length, 512]);
    const scores = this.model.predict(input) as tf.Tensor;
    const scoreArray = scores.dataSync() as Float32Array;

    // En önemli cümleleri seç
    const scoredSentences = sentences.map((s, i) => ({ sentence: s, score: scoreArray[i] }));
    const sortedSentences = scoredSentences.sort((a, b) => b.score - a.score).slice(0, Math.min(maxLength / 10, sentences.length));

    // Özet oluştur
    const summary = sortedSentences.map((s) => s.sentence.trim()).join(". ").substring(0, maxLength) + ".";
    return summary;
  }

  // Modeli eğit (örnek veriyle)
  async trainSummarizerModel(trainingData: { text: string; summary: string }[]) {
    if (!this.model || !this.encoder) throw new Error("Model veya encoder başlatılmadı");

    const inputs = [];
    const labels = [];
    for (const data of trainingData) {
      const sentences = data.text.split(/[.!?]+/).filter((s) => s.trim().length > 0).slice(0, 10);
      const embeddings = await this.encoder.embed(sentences);
      const input = embeddings.arraySync()[0] as number[]; // İlk cümleyi örnek al
      inputs.push(input);

      const targetSentences = data.summary.split(/[.!?]+/).filter((s) => s.trim().length > 0).slice(0, 5);
      const targetEmbeddings = await this.encoder.embed(targetSentences);
      const label = tf.mean(targetEmbeddings, 0).dataSync() as number[]; // Ortalama vektör
      labels.push(label);
    }

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 512]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 512]);

    await this.model.fit(inputTensor, labelTensor, {
      epochs: 200,
      batchSize: 32,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}, mae = ${log.mae}`),
      },
    });

    await this.model.save("file://./summarizerModel");
    console.log("Metin özetleme modeli eğitimi tamamlandı ve kaydedildi.");
  }
}

// Örnek kullanım
const textSummarizer = new TextSummarizer();
textSummarizer.summarizeText("Bu uzun bir bilimsel makale, burada bilimsel bir keşif yaptık ve sonuçlar çok etkileyici. Deneylerimizi detaylı bir şekilde analiz ettik ve harika sonuçlar elde ettik.", 50).then(console.log);