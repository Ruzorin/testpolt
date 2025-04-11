import * as tf from "@tensorflow/tfjs";
import * as speechCommands from "@tensorflow-models/speech-commands";

// Ses analizi modeli
export class AudioAnalyzer {
  private model: speechCommands.SpeechCommandRecognizer | null = null;

  constructor() {
    this.initializeModel();
  }

  private async initializeModel() {
    this.model = speechCommands.create("BROWSER_FFT");
    await this.model.ensureModelLoaded();
  }

  async analyzeAudio(audioBlob: Blob): Promise<{
    sentiment: "positive" | "neutral" | "negative";
    keywords: string[];
    isAppropriate: boolean;
  }> {
    if (!this.model) throw new Error("Model başlatılmadı");

    // Audio’yu yükle ve analiz et
    const audioContext = new AudioContext();
    const arrayBuffer = await audioBlob.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    const samples = audioBuffer.getChannelData(0);

    // Ses verilerini Tensor’a dönüştür
    const input = tf.tensor1d(samples).reshape([samples.length, 1]);
    const predictions = await this.model.recognize(input);

    // Duygu analizi (örnek basit mantık)
    const sentimentKeywords = {
      positive: ["happy", "good", "great"],
      neutral: ["ok", "normal", "fine"],
      negative: ["bad", "sad", "terrible"],
    };
    let sentiment: "positive" | "neutral" | "negative" = "neutral";
    let maxScore = 0;
    for (const [sent, keywords] of Object.entries(sentimentKeywords)) {
      const score = predictions.reduce((sum, pred) => sum + (keywords.includes(pred.label.toLowerCase()) ? pred.probability : 0), 0);
      if (score > maxScore) {
        maxScore = score;
        sentiment = sent as "positive" | "neutral" | "negative";
      }
    }

    // Anahtar kelimeler ve uygunluk kontrolü
    const keywords = predictions
      .filter((pred) => pred.probability > 0.5)
      .map((pred) => pred.label)
      .slice(0, 5);
    const inappropriateKeywords = ["violence", "hate", "curse"];
    const isAppropriate = !keywords.some((kw) => inappropriateKeywords.includes(kw.toLowerCase()));

    return { sentiment, keywords, isAppropriate };
  }

  // Ses verisiyle modeli eğit (örnek)
  async trainAudioModel(trainingData: { audioBlob: Blob; sentiment: "positive" | "neutral" | "negative"; isAppropriate: boolean }[]) {
    if (!this.model) throw new Error("Model başlatılmadı");

    const inputs = [];
    const labels = [];
    for (const data of trainingData) {
      const audioContext = new AudioContext();
      const arrayBuffer = await data.audioBlob.arrayBuffer();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      const samples = audioBuffer.getChannelData(0);
      inputs.push(tf.tensor1d(samples).reshape([samples.length, 1]).dataSync());

      const labelVector = [0, 0, 0];
      const sentimentIndex = ["negative", "neutral", "positive"].indexOf(data.sentiment);
      if (sentimentIndex !== -1) labelVector[sentimentIndex] = 1;
      labels.push(labelVector);
    }

    const inputTensor = tf.tensor3d(inputs, [inputs.length, inputs[0].length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 3]);

    // Özel bir model oluştur ve eğit
    const customModel = tf.sequential();
    customModel.add(tf.layers.conv1d({ filters: 64, kernelSize: 5, activation: "relu", inputShape: [inputs[0].length, 1] }));
    customModel.add(tf.layers.maxPooling1d({ poolSize: 2 }));
    customModel.add(tf.layers.flatten());
    customModel.add(tf.layers.dense({ units: 32, activation: "relu" }));
    customModel.add(tf.layers.dense({ units: 3, activation: "softmax" }));
    customModel.compile({ optimizer: "adam", loss: "categoricalCrossentropy", metrics: ["accuracy"] });

    await customModel.fit(inputTensor, labelTensor, {
      epochs: 100,
      batchSize: 16,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}, accuracy = ${log.acc}`),
      },
    });

    await customModel.save("file://./audioModel");
    console.log("Ses analizi modeli eğitimi tamamlandı ve kaydedildi.");
  }
}

// Örnek kullanım
const audioAnalyzer = new AudioAnalyzer();
fetch("https://example.com/audio.mp3")
  .then((response) => response.blob())
  .then((blob) => audioAnalyzer.analyzeAudio(blob))
  .then(console.log);