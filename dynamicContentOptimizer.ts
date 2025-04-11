import * as tf from "@tensorflow/tfjs";
import * as use from "@tensorflow-models/universal-sentence-encoder";
import { Server } from "socket.io";
import { Post } from "../server/models/Post";
import { UserBehavior } from "./behaviorPredictor";
import { ContentOptimizer } from "./contentOptimizer";

// Gerçek zamanlı içerik optimizasyonu
export class DynamicContentOptimizer {
  private model: ContentOptimizer;
  private io: Server;
  private users: Map<string, UserBehavior> = new Map();
  private contentCache: Map<string, Post> = new Map();

  constructor(io: Server) {
    this.io = io;
    this.model = new ContentOptimizer();
    this.initializeListeners();
  }

  private initializeListeners() {
    this.io.on("connection", (socket) => {
      console.log("Kullanıcı bağlandı:", socket.id);

      socket.on("userBehavior", async (behavior: Partial<UserBehavior>) => {
        const userAddress = behavior.address || socket.id;
        let fullBehavior = this.users.get(userAddress) || { address: userAddress, viewedPosts: [], likedPosts: [], selectedCategories: [], interactions: [], lastLogin: Date.now(), sessionDuration: 0 };
        fullBehavior = { ...fullBehavior, ...behavior, lastLogin: Date.now(), sessionDuration: (fullBehavior.sessionDuration || 0) + 1 };

        this.users.set(userAddress, fullBehavior);
        this.optimizeContentForUser(userAddress, fullBehavior);
      });

      socket.on("newContent", async (content: Partial<Post>) => {
        const postId = content.id || `${Date.now()}-${Math.random().toString(36).slice(2)}`;
        this.contentCache.set(postId, content as Post);
        this.optimizeAndBroadcastContent(postId);
      });

      socket.on("disconnect", () => {
        this.users.delete(userAddress);
        console.log("Kullanıcı ayrıldı:", socket.id);
      });
    });
  }

  private async optimizeContentForUser(userAddress: string, behavior: UserBehavior) {
    for (const [postId, post] of this.contentCache) {
      const optimization = await this.model.optimizeContent(post.content, behavior);
      this.io.to(userAddress).emit("contentOptimization", { postId, ...optimization });
    }
  }

  private async optimizeAndBroadcastContent(postId: string) {
    const post = this.contentCache.get(postId);
    if (!post) return;

    for (const [userAddress, behavior] of this.users) {
      const optimization = await this.model.optimizeContent(post.content, behavior);
      this.io.to(userAddress).emit("contentOptimization", { postId, ...optimization });
    }
  }

  // Modeli güncelle ve eğit (gerçek zamanlı veriyle)
  async updateAndTrainModel(newData: { content: string; userBehavior: Partial<UserBehavior>; performance: "high" | "medium" | "low" }[]) {
    await this.model.trainContentOptimizer(newData);
    console.log("İçerik optimizasyon modeli gerçek zamanlı olarak güncellendi ve eğitildi.");
  }
}

// Örnek kullanım (server tarafında)
import { createServer } from "http";
import { Server } from "socket.io";
const httpServer = createServer();
const io = new Server(httpServer);
const dynamicContentOptimizer = new DynamicContentOptimizer(io);
httpServer.listen(3002, () => console.log("Dynamic content optimizer server çalışıyor: 3002"));