import pandas as pd
import pyperclip as pc
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def denormalize(item, list):
	return (item * (max(list) - min(list)) + min(list))
	
def main():
	try:
		data = pd.read_csv('data.csv')

		dataS = pd.DataFrame(MinMaxScaler().fit_transform(data))
		X = dataS.iloc[:, 0]
		Y = dataS.iloc[:, 1]

		slope = 0
		intercept = 0
		L = 0.1
		epochs = 1000

		m = float(len(X))

		for i in range(epochs):
			Y_pred = intercept + slope * X

			D_intercept = sum(Y_pred - Y) / m
			D_slope = sum(X * (Y_pred - Y)) / m

			slope -= L * D_slope
			intercept -= L * D_intercept

			print('Regression Line -> Price = {} {} * mileage'.format(
				f'{intercept:.5}',
				slope/100
				# f'{abs(slope):.5f}'
			))

		y_minmax = [7994.025903991332 - 0.009953762249792933 * x for x in data['km']]
		fig, ax = plt.subplots()
		ax.scatter(data['km'], data['price'])
		ax.plot(data['km'], y_minmax)
		plt.show()
			# if i % 10 == 0:
			# 	cost = (1/(2*m)) * sum((Y - Y_pred)**2)
			# 	print(cost)
		
		pc.copy(f'{denormalize(intercept, data['price'])} {'+' if slope > 0 else '-'} {abs(slope)/100}')
		print('Values have been saved to clipboard! (intercept, slope)')

	except Exception as e:
		print(f'{type(e).__name__} : {e}')

if __name__ == "__main__":
	main()