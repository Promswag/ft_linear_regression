import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class LinearRegression():
	def __init__(self, X: np.ndarray[float, int], Y: np.ndarray[float, int], learning_rate:float=0.1, epochs:int=2000):
		try:
			if len(X) != len(Y): 
				raise ValueError("LinearRegression: X and Y must be equal length.")
			if len(X) < 3:
				raise ValueError("LinearRegression: X and Y must be of length of 3 or more.")
			for x in X:
				if not isinstance(x, (float, int)):
					raise ValueError("LinearRegression: X must contain float or int only.")
			for y in Y:
				if not isinstance(y, (float, int)):
					raise ValueError("LinearRegression: Y must contain float or int only.")
			self.m = float(len(X))
			self.X = X
			self.Y = Y
			self.X_n = np.array([self.normalize(x, X) for x in X])
			self.Y_n = np.array([self.normalize(y, Y) for y in Y])
			self.learning_rate = learning_rate
			self.epochs = epochs
			self.theta0 = 0
			self.theta1 = 0
			self.costs = []
		except Exception as e:
			print(f'{type(e).__name__}: {e}')
			return None
	
	def reset(self):
		self.theta0 = 0
		self.theta1 = 0
		self.costs.clear()

	@staticmethod
	def normalize(item, list):
		return ((item - min(list)) / (max(list) - min(list)))
	
	@staticmethod
	def denormalize(item, list):
		return (item * (max(list) - min(list)) + min(list))
	
	def gradient_descent(self, save_cost: bool=False):
		for _ in range(self.epochs):
			Y_pred = self.theta0 + self.theta1 * self.X_n
			if save_cost is True:
				self.costs.append([self.theta0, sum((self.Y_n - Y_pred)**2)])
			self.theta0 -= self.learning_rate * sum(Y_pred - self.Y_n) / self.m
			self.theta1 -= self.learning_rate * sum(self.X_n * (Y_pred - self.Y_n)) / self.m

	def plot(self, figsize: tuple[int | float]=(12, 6), xlabel: str=None, ylabel: str=None):
		fig, axes = plt.subplots(ncols=2, figsize=figsize)

		axes[0].scatter(self.X_n, self.Y_n)
		axes[0].plot(
			(0, 1),
			(self.theta0, self.theta0 + self.theta1)
		)
		axes[0].set_title("With normalized data")
		axes[0].set_facecolor('#E0E0E0')

		axes[1].scatter(self.X, self.Y)
		axes[1].plot(
			(self.denormalize(0, self.X), self.denormalize(1, self.X)),
			(self.denormalize(self.theta0, self.Y), self.denormalize(self.theta0 + self.theta1, self.Y))
		)
		axes[1].set_title("With input data")
		axes[1].set_facecolor('#E0E0E0')
	
		plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
		plt.show()	
	
	def cost(self, figsize: tuple[int | float]=(12, 6)):
		if len(self.costs) == 0:
			print("LinearRegression: No cost to plot.")
			return
		fig, ax = plt.subplots(figsize=figsize)

		ax.plot(np.array(self.costs)[:,0], np.array(self.costs)[:,1])
		ax.set_title("With normalized data")
		ax.set_xlabel("theta0 (Intercept)")
		ax.set_ylabel("Sum of squared residuals (SSR)")
		ax.set_facecolor('#E0E0E0')
	
		plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
		plt.show()
	
def main():
	try:

		data = pd.read_csv('data.csv')
		X = data.iloc[:, 0]
		Y = data.iloc[:, 1]

		lr = LinearRegression(X, Y)
		lr.gradient_descent(save_cost=True)
		lr.plot()
		lr.cost()

	except Exception as e:
		print(f'{type(e).__name__} : {e}')

if __name__ == "__main__":
	main()