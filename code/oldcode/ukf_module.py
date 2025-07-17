import numpy as np
import math

class UnscentedKalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise_cov, measurement_noise_cov, alpha=0.1, beta=2):
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_noise_cov = process_noise_cov
        self.measurement_noise_cov = measurement_noise_cov
        self.alpha = alpha
        self.beta = beta

        # 计算西格玛点的参数
        self.n = len(initial_state)
        self.kappa = 0
        self.lambda_ = self.n + self.kappa

        # 计算权重
        self.Wm = np.zeros(2 * self.n + 1)
        self.Wc = np.zeros(2 * self.n + 1)
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.n + self.lambda_) + (1 - math.pow(self.alpha, 2) + self.beta)
        for i in range(1, 2 * self.n + 1):
            self.Wm[i] = 1 / (2 * (self.n + self.lambda_))
            self.Wc[i] = 1 / (2 * (self.n + self.lambda_))

    def f(self,state, control_input):
        # 这里简单假设一个非线性运动模型
        x = state[0] + control_input[0] * np.cos(state[1])
        y = state[1] + control_input[0] * np.sin(state[1])
        return np.array([x, y])

    def h(self,state):
        # 假设测量模型为直接返回状态
        return state

    def generate_sigma_points(self):
        # 计算平方根矩阵
        root_cov = np.linalg.cholesky((self.n + self.lambda_) * self.covariance)
        # 生成西格玛点
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sigma_points[0] = self.state
        for i in range(self.n):
            sigma_points[i + 1] = self.state + root_cov[:, i]
            sigma_points[self.n + i + 1] = self.state - root_cov[:, i]
        return sigma_points

    def predict(self, control_input):
        # 生成西格玛点
        sigma_points = self.generate_sigma_points()
        # 传播西格玛点
        propagated_sigma_points = np.array([self.f(point, control_input) for point in sigma_points])
        # 计算预测状态
        predicted_state = np.dot(self.Wm, propagated_sigma_points)
        # 计算预测协方差
        predicted_covariance = np.zeros((self.n, self.n))
        for i in range(2 * self.n + 1):
            diff = propagated_sigma_points[i] - predicted_state
            predicted_covariance += self.Wc[i] * np.outer(diff, diff)
        predicted_covariance += self.process_noise_cov
        self.state = predicted_state
        self.covariance = predicted_covariance

    def update(self, measurement):
        # 生成预测的西格玛点
        predicted_sigma_points = self.generate_sigma_points()
        # 计算预测的测量值
        predicted_measurements = np.array([self.h(point) for point in predicted_sigma_points])
        # 计算测量的均值
        predicted_measurement_mean = np.dot(self.Wm, predicted_measurements)
        # 计算创新协方差
        innovation_covariance = np.zeros((len(measurement), len(measurement)))
        for i in range(2 * self.n + 1):
            diff = predicted_measurements[i] - predicted_measurement_mean
            innovation_covariance += self.Wc[i] * np.outer(diff, diff)
        innovation_covariance += self.measurement_noise_cov
        # 计算交叉协方差
        cross_covariance = np.zeros((self.n, len(measurement)))
        for i in range(2 * self.n + 1):
            diff_state = predicted_sigma_points[i] - self.state
            diff_measurement = predicted_measurements[i] - predicted_measurement_mean
            cross_covariance += self.Wc[i] * np.outer(diff_state, diff_measurement)
        # 计算卡尔曼增益
        kalman_gain = np.dot(cross_covariance, np.linalg.inv(innovation_covariance))
        # 更新状态估计
        innovation = measurement - predicted_measurement_mean
        self.state = self.state + np.dot(kalman_gain, innovation)
        # 更新协方差估计
        self.covariance = self.covariance - np.dot(np.dot(kalman_gain, innovation_covariance), kalman_gain.T)