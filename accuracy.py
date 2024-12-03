from LinearRegression import LinearRegression as LR
import pandas as pd

def main():
	try:

		data = pd.read_csv('data.csv')
		X = data.iloc[:, 0]
		Y = data.iloc[:, 1]

		with open('thetas', 'r') as file:
			content = file.readline().split(',')
			if len(content) != 2:
				print('Thetas file corrupted, set thetas:')
				theta0 = float(input('theta0: '))
				theta1 = float(input('theta1: '))
			else:
				theta0 = float(content[0])
				theta1 = float(content[1])

		data['estimated_price'] = theta0 + theta1 * data['km']
		MAE = sum(abs(data['price'] - data['estimated_price'])) / len(data)
		print(MAE)
		return

		data['diff'] = abs(data['price'] - data['estimated_price'])
		data['ratio'] = (100 / data['price']) * data['diff']
		print(data['ratio'].mean())

		
	except Exception as e:
		print(f'{type(e).__name__} : {e}')

if __name__ == "__main__":
	main()