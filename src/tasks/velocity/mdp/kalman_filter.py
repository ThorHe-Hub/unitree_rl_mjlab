import numpy as np

class KalmanFilter:
    def __init__(self, dt=0.02):
        self.dt = dt
        # 状态向量 X = [位置p, 速度v, 加速度a]^T
        self.X = np.zeros((3, 1))
        
        # 恒加速度(CA)状态转移矩阵 A
        self.A = np.array([
            [1.0, self.dt, 0.5 * self.dt**2],
            [0.0, 1.0,     self.dt],
            [0.0, 0.0,     1.0]
        ])
        
        # 观测矩阵 H (仅直接观测速度 v)
        self.H = np.array([[0.0, 1.0, 0.0]])
        
        self.P = np.eye(3) * 1.0
        
        # 过程噪声协方差 Q 
        self.Q = np.diag([0.01, 0.01, 0.0005])
        
        # 测量噪声协方差 R
        self.R = np.array([[0.25]])

    def predict(self):
        self.X = np.dot(self.A, self.X)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        # 计算测量残差
        innovation = z - np.dot(self.H, self.X)[0, 0]
        
        # 卡尔曼增益更新
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), 1.0 / S[0, 0])
        
        self.X = self.X + K * innovation
        I = np.eye(3)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        
        # 返回滤波后的速度分量
        return self.X[1, 0].item()