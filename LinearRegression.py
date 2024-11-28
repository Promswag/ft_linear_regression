import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class LinearRegression():
	def __init__(self, X: np.ndarray[float, int], Y: np.ndarray[float, int], learning_rate: float=0.1, epochs: int=2000):
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

	def show_data(self, figsize: tuple[int | float]=(12, 6), xlabel: str=None, ylabel: str=None):
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
	
	def show_cost(self, figsize: tuple[int | float]=(12, 6)):
		if len(self.costs) == 0:
			print("LinearRegression: No cost to plot.")
			return
		fig, ax = plt.subplots(figsize=figsize)

		ax.plot(np.array(self.costs)[:,0], np.array(self.costs)[:,1])
		ax.set_title("Gradient Descent")
		ax.set_xlabel("Intercept (theta0)")
		ax.set_ylabel("Sum of squared residuals (SSR)")
		ax.set_facecolor('#E0E0E0')
	
		plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
		plt.show()

	def gradient_descent_step(self) -> tuple[float, float]:
		Y_pred = self.theta0 + self.theta1 * self.X_n
		self.theta0 -= self.learning_rate * sum(Y_pred - self.Y_n) / self.m
		self.theta1 -= self.learning_rate * sum(self.X_n * (Y_pred - self.Y_n)) / self.m
		return self.theta0, self.theta1
	
	def gradient_descent_step_cost(self):
		self.costs.append([self.theta0, sum((self.Y_n - (self.theta0 + self.theta1 * self.X_n))**2)])
	
	def realtime(self, learning_rate: float=0.1, figsize: tuple[int | float]=(12, 6)):
		self.learning_rate = learning_rate
		fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=figsize)
		ax1.scatter(self.X_n, self.Y_n)
		ax1.set_xlim(-0.05, 1.05)
		ax1.set_ylim(-0.05, 1.05)
		ax2.set_xlim(-0.05, 1.05)
		ax2.set_ylim(-0.05, 10.05)
		ax1.set_facecolor('#E0E0E0')
		ax2.set_facecolor('#E0E0E0')
		ax3.set_facecolor('#E0E0E0')
		ax3.set_xlim(-50,1000)
		ax3.set_ylim(0,1)
		line1, = ax1.plot([], [])
		line2, = ax2.plot([], [])
		line3, = ax3.plot([], [])
		line4, = ax3.plot([], [])

		mse = []
		rs = []
		def init():
			line1.set_data([], [])
			line2.set_data([], [])
			line3.set_data([], [])
			line4.set_data([], [])
			return line1, line2, line3, line4

		def animate(i):
			theta0, theta1 = self.gradient_descent_step()
			line1.set_data((0, 1), (theta0, theta0 + theta1))

			self.gradient_descent_step_cost()
			line2.set_data(np.array(self.costs)[:,0], np.array(self.costs)[:,1])
			mse.append(sum((self.Y_n - (theta0 + theta1 * self.X_n)**2)) / self.m)
			rs.append(1 - (sum((self.Y_n - (theta0 + theta1 * self.X_n))**2) / sum((self.Y_n - self.Y_n.mean())**2)))
			line3.set_data(range(len(mse)), mse)
			line4.set_data(range(len(rs)), rs)
			# print(f"MSE: {mse} | R²: {rs}")

			if i % 50 == 0:
				stop_animation_trigger()
			return line1, line2, line3, line4
		
		def stop_animation_trigger():
			if abs(sum(-2 * (self.Y_n - (self.theta0 + self.theta1 * self.X_n)))) < 0.01:
				ani.event_source.stop()

		ani = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=0.1, blit=True)
		plt.show()

	def predict(self, x_value: float | int, theta0: float=None, theta1: float=None):
		if theta0 is None:
			theta0 = self.theta0
		if theta1 is None:
			theta1 = self.theta1
		return theta0 + theta1 * x_value

	@staticmethod
	def predict(x_value: float | int, theta0: float, theta1: float) -> float:
		try:
			with open("thetas") as file:
				theta0 = file.readline()
				theta1 = file.readline()
				return theta0 + theta1 * x_value
		except Exception as e:
			print(f'{type(e).__name__}: {e}')
			return None