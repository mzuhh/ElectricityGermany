import pandas as pd


#Dynamic Directories
if os.getenv('PYTHON_ENV') == 'pycharm':
    data_dir = 'C:/Users/MichalZlotnik/PycharmProjects/ElectricityGermany/data'
else:
    data_dir = '/content/ElectricityGermany/data'

file_name = 'day_ahead_price_germany.csv'
file_path = os.path.join(data_dir, file_name)
print(f"Loading data from {file_path}")


df = pd.read_csv(file_path, index_col=date_column_name)



