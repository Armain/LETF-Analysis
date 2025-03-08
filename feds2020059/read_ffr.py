from bs4 import BeautifulSoup
import pandas as pd

# Read the HTML file
with open('D:/Schoolwork - McGill/Other/LETF Analysis/feds2020059/accessible_figures.html', 'r') as f:
    html_content = f.read()

soup = BeautifulSoup(html_content, 'html.parser')
tables = soup.find_all('table')

def extract_table_to_df(table):
    """Extract table data and convert to DataFrame"""
    data = []
    for row in table.find_all('tr'):
        cols = row.find_all(['td', 'th'])
        cols = [col.text.strip() for col in cols]
        if cols:  # Only append non-empty rows
            data.append(cols)
    df = pd.DataFrame(data[1:], columns=data[0])  # First row as headers
    df = df[['Date', 'Fed Funds Rate']]
    return df

# Extract each table
df1 = extract_table_to_df(tables[0])  # 1928-1933
df2 = extract_table_to_df(tables[3])  # 1933-1940
df3 = extract_table_to_df(tables[5])  # 1940-1954

historical_ffr = pd.concat([df1, df2, df3], ignore_index=True)
historical_ffr = historical_ffr.drop_duplicates()
historical_ffr = historical_ffr.sort_values('Date')
historical_ffr['Date'] = pd.to_datetime(historical_ffr['Date'])
historical_ffr.set_index('Date', inplace=True)
historical_ffr.to_csv('D:/Schoolwork - McGill/Other/LETF Analysis/ffr_1928_1954.csv', index=False)
historical_ffr.rename(columns={'Fed Funds Rate': 'value', 'Date': 'DATE'}, inplace=True)
historical_ffr = historical_ffr.iloc[:-1]
historical_ffr['value'] = pd.to_numeric(historical_ffr['value'], errors='coerce')

# Remove any rows where value is NaN (if any were created by the conversion)
historical_ffr = historical_ffr.dropna(subset=['value'])
# historical_ffr.plot()

ffr_file_path = 'D:/Schoolwork - McGill/Other/LETF Analysis/ffr_1954_2024.csv'
ffr = pd.read_csv(ffr_file_path, index_col='DATE', parse_dates=True)
# ffr.plot()

ffr = pd.concat([historical_ffr, ffr])
ffr.plot()

ffr.to_csv('D:/Schoolwork - McGill/Other/LETF Analysis/ffr.csv', index=True)
