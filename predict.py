from LinearRegression import LinearRegression as LR
import os


def main():
    try:
        theta0 = 0
        theta1 = 0
        path = 'thetas'
        if os.path.isfile(path):
            with open(path, 'r') as file:
                content = file.readline().split(',')
                if len(content) != 2:
                    print('Thetas file corrupted')
                    return
                else:
                    theta0 = float(content[0])
                    theta1 = float(content[1])
        mileage = input('Mileage of the car: ')
        mileage = float(mileage)
        print(f'Estimated price: {LR.predict(mileage, theta0=theta0, theta1=theta1)}')
    except Exception as e:
        print(f'{type(e).__name__} : {e}')


if __name__ == "__main__":
    main()
