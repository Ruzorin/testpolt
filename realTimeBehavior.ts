import * as tf from "@tensorflow/tfjs";
import { Server } from "socket.io";
import { UserBehavior } from "./behaviorPredictor";
import { BehaviorPredictor } from "./behaviorPredictor";

// Gerçek zamanlı davranış analizi
export class RealTimeBehavior {
  private model: BehaviorPredictor;
  private io: Server;
  private users: Map<string, UserBehavior> = new Map();

  constructor(io: Server) {
    this.io = io;
    this.model = new BehaviorPredictor();
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
        const prediction = await this.model.predictBehavior(fullBehavior);
        socket.emit("behaviorUpdate", { userAddress, ...prediction });

        // Gerçek zamanlı güncellemeler için 5 saniyede bir kontrol
        const interval = setInterval(async () => {
          const updatedPrediction = await this.model.predictBehavior(fullBehavior);
          socket.emit("behaviorUpdate", { userAddress, ...updatedPrediction });
        }, 5000);

        socket.on("disconnect", () => {
          clearInterval(interval);
          this.users.delete(userAddress);
          console.log("Kullanıcı ayrıldı:", socket.id);
        });
      });
    });
  }

  // Modeli güncelle ve eğit (gerçek zamanlı veriyle)
  async updateAndTrainModel(newData: { user: UserBehavior; engagement: "high" | "medium" | "low" }[]) {
    await this.model.trainBehaviorModel(newData);
    console.log("Davranış tahmini modeli gerçek zamanlı olarak güncellendi ve eğitildi.");
  }

  // Kullanıcı davranışını gerçek zamanlı analiz et
  async analyzeBehaviorInRealTime(userAddress: string): Promise<{ engagementLevel: "high" | "medium" | "low"; score: number }> {
    const behavior = this.users.get(userAddress);
    if (!behavior) throw new Error("Kullanıcı davranışı bulunamadı");

    return await this.model.predictBehavior(behavior);
  }
}

// Örnek kullanım (server tarafında)
import { createServer } from "http";
import { Server } from "socket.io";
const httpServer = createServer();
const io = new Server(httpServer);
const realTimeBehavior = new RealTimeBehavior(io);
httpServer.listen(3001, () => console.log("Real-time behavior server çalışıyor: 3001"));