from LinearRegression import LinearRegression as LR
import pandas as pd

def main():
	try:

		data = pd.read_csv('data.csv')
		X = data.iloc[:, 0]
		Y = data.iloc[:, 1]

		lr = LR(X, Y)
		lr.gradient_descent(save_cost=True)
		lr.show_data()
		lr.show_cost()
		lr.reset()
		lr.realtime()

	except Exception as e:
		print(f'{type(e).__name__} : {e}')

if __name__ == "__main__":
	main()