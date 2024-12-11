import matplotlib.pyplot as plt
import pandas as pd


def main():
    try:
        data = pd.read_csv('data.csv')
        with open('thetas', 'r') as file:
            content = file.readline().split(',')
            if len(content) != 2:
                print('Thetas file corrupted, set thetas:')
                theta0 = float(input('theta0: '))
                theta1 = float(input('theta1: '))
            else:
                theta0 = float(content[0])
                theta1 = float(content[1])

        mean = data['price'].mean()
        data['estimated_price'] = theta0 + theta1 * data['km']
        MAE = sum(abs(data['price'] - data['estimated_price'])) / len(data)
        print('MAE €: {}\nMAE %: {}'.format(
            f'{MAE:.2f}',
            f'{(100 / mean * MAE):.2f}'
        ))

        data['ratio'] = (100 / data['price']) * abs(data['price'] - data['estimated_price'])
        data['diff'] = abs(data['price'] - data['estimated_price'])
        data['abs'] = data['price'] + data['diff']

        x = range(len(data))
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))

        ax1.set_title("Estimated Prices vs Real Prices (Ratio %)")
        ax1.plot(x, data['ratio'])
        ax1.axhline(y=data['ratio'].mean(), color='r', linestyle='--', label='Ratio Average')
        ax1.axhline(y=100 / mean * MAE, color='g', linestyle='--', label='Ratio Average (agg.)')
        ax1.set_xlim(0, len(x) - 1)
        ax1.set_ylim(0, max(data['ratio']) * 1.1)
        ax1.legend(loc='upper right')

        ax2.set_title("Real Prices vs Absolute Error")
        ax2.fill_between(x, 0, data['price'], color='b', alpha=0.5, label='Real Prices')
        ax2.fill_between(x, data['price'], data['abs'], color='r', alpha=0.5, label='Absolute Error')
        ax2.set_xlim(0, len(x) - 1)
        ax2.set_ylim(0, max(data['abs'] * 1.1))
        ax2.legend(loc='upper left')

        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.25)
        plt.show()

    except Exception as e:
        print(f'{type(e).__name__} : {e}')


if __name__ == "__main__":
    main()
