

def main():
	try:
		mileage = input('Mileage of the car: ')
		mileage = float(mileage)
		price = 7994.025903991332 - 0.009953762249792932 * mileage
		print(price)
		pass
	except Exception as e:
		print(f'{type(e).__name__} : {e}')

if __name__ == "__main__":
	main()