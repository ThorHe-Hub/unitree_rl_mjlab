import numpy as np

class AdaptiveKalmanfilter:
    def __init__(self, dt=0.02):
        self.dt = dt
        self.X = np.zeros((3, 1))
        
        self.A = np.array([
            [1.0, self.dt, 0.5 * self.dt**2],
            [0.0, 1.0,     self.dt],
            [0.0, 0.0,     1.0]
        ])
        
        self.H = np.array([[0.0, 1.0, 0.0]])
        self.P = np.eye(3) * 1.0
        
        self.Q = np.diag([0.01, 0.01, 0.001])
        self.R = np.array([[0.25]])
        self.b = 0.97
        self.step_count = 0
        
        self.R_min, self.R_max = 0.05, 2.0
        self.Q_min, self.Q_max = 0.001, 0.05

    def predict(self):
        self.X = np.dot(self.A, self.X)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        """线速度 Vx 专用：常规双自适应"""
        self.step_count += 1
        residual = z - np.dot(self.H, self.X)[0, 0]
        H_P_HT = np.dot(np.dot(self.H, self.P), self.H.T)[0, 0]
        
        d_k = (1 - self.b) / (1 - self.b**(min(self.step_count, 50)))
        innovation_energy = residual**2
        
        R_est = (1 - d_k) * self.R[0, 0] + d_k * max(0, innovation_energy - H_P_HT)
        self.R[0, 0] = np.clip(R_est, self.R_min, self.R_max)
        
        S = H_P_HT + self.R[0, 0]
        K = np.dot(self.P, self.H.T) / S
        
        K_vec = K.reshape(3)
        for i in range(3):
            q_est = (1 - d_k) * self.Q[i, i] + d_k * (K_vec[i]**2 * innovation_energy)
            self.Q[i, i] = np.clip(q_est, self.Q_min, self.Q_max)
            
        self.X = self.X + K * residual
        self.P = np.dot((np.eye(3) - np.dot(K, self.H)), self.P)
        
        return self.X[1, 0].item()

    def update_for_yaw(self, z):
        """角速度 Yaw 专用：抗震荡死锁自适应"""
        self.step_count += 1
        residual = z - np.dot(self.H, self.X)[0, 0]
        H_P_HT = np.dot(np.dot(self.H, self.P), self.H.T)[0, 0]
        
        # 1. 钳位残差能量，防止高频震荡导致 R 瞬间发散
        innovation_energy = np.clip(residual**2, 0.0, 0.5)
        
        d_k = (1 - self.b) / (1 - self.b**(min(self.step_count, 50)))
        
        # 2. 压低 R_max，上限设为 0.2，严禁滤波器“自闭”拒收观测数据
        R_est = (1 - d_k) * self.R[0, 0] + d_k * max(0, innovation_energy - H_P_HT)
        self.R[0, 0] = np.clip(R_est, 0.01, 0.2)
        
        S = H_P_HT + self.R[0, 0]
        K = np.dot(self.P, self.H.T) / S
        
        # 3. 注入高频动态宽容度，强制拉高加速度的 Q 权重，防止 P 矩阵塌缩
        self.Q = np.diag([0.05, 0.1, 0.4])
        
        self.X = self.X + K * residual
        self.P = np.dot((np.eye(3) - np.dot(K, self.H)), self.P)
        
        return self.X[1, 0].item()