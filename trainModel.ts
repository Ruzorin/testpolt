import * as tf from "@tensorflow/tfjs";
import { Post } from "../server/models/Post";
import { UserBehavior } from "./recommendationEngine";

// Eğitim verisi (örnek)
const trainingData: { user: UserBehavior; postIds: string[] }[] = [
  {
    user: {
      address: "0x123",
      viewedPosts: ["1"],
      likedPosts: ["2"],
      selectedCategories: ["Bilim"],
      interactions: [{ postId: "1", timestamp: Date.now(), action: "view" }],
    },
    postIds: ["1", "2"],
  },
  {
    user: {
      address: "0x456",
      viewedPosts: ["2"],
      likedPosts: [],
      selectedCategories: ["Futbol"],
      interactions: [{ postId: "2", timestamp: Date.now(), action: "view" }],
    },
    postIds: ["2"],
  },
];

const posts: Post[] = [
  {
    id: "1",
    type: "video",
    content: "Bilim Videosu",
    category: "Bilim",
    creator: "0x123",
    likes: 10,
    timestamp: "2025-02-24",
    _id: new mongoose.Types.ObjectId().toString(), // Mongoose _id simülasyonu
    $assertPopulated: () => {}, // Boş bir fonksiyon, Mongoose için gereksinim
    $clearModifiedPaths: () => {}, // Boş bir fonksiyon
    $clone: () => ({}) as any, // Boş bir fonksiyon
    // Diğer Mongoose yöntemlerini boş tut, yalnızca TypeScript uyumu için
  },
  {
    id: "2",
    type: "tweet",
    content: "Maç yorumu",
    category: "Futbol",
    creator: "0x456",
    likes: 5,
    timestamp: "2025-02-24",
    _id: new mongoose.Types.ObjectId().toString(),
    $assertPopulated: () => {},
    $clearModifiedPaths: () => {},
    $clone: () => ({}) as any,
  },
];

async function trainAdvancedModel() {
  // Öneri motoru modelini kullan
  const recommendationEngine = new (await import("./recommendationEngine")).RecommendationEngine();
  await recommendationEngine.trainRecommendationModel(trainingData);

  // Spam filtreleme modelini eğit
  const spamFilter = new (await import("./spamFilter")).SpamFilter();
  const spamTrainingData = [
    { text: "Ücretsiz para kazan, hızlı zengin ol!", isSpam: true },
    { text: "Merhaba, güzel bir gün!", isSpam: false },
  ];
  await spamFilter.trainSpamModel(spamTrainingData);

  console.log("Tüm modeller eğitimi tamamlandı.");
}

trainAdvancedModel().catch((error) => {
  console.error("Model eğitimi başarısız:", error);
});