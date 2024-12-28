Pandas
What is Pandas?
Pandas is a popular open-source data analysis and manipulation library, built on top of the Python programming language. It provides flexible and efficient data structures that make it easier to handle and analyze data. Pandas is particularly well-suited for structured or tabular data, such as CSV and Excel files, SQL databases, or dataframes in Python.

Why use Pandas for Data Engineering?
Data Handling: Pandas can handle a variety of data sets in different formats - CSV files, Excel files, or database records.

Ease of Use: With only a few lines of code, Pandas makes it easy for users to read, write, and modify datasets.

Data Transformation: It offers robust tools for cleaning and pivoting data, preparing it for analysis or visualization.

Efficient Operations: Pandas is built on top of NumPy, a Python library for numerical computation, which makes it efficient for performing operations on large datasets.

Integration: It integrates well with many other libraries in the scientific Python ecosystem, such as Matplotlib for plotting graphs, Scikit-learn for machine learning, and many others.

Installing Pandas
Pandas can be installed in your Python environment using package managers like pip or conda.

Installing Pandas using pip:
If you’re using a Jupyter notebook, you can install it using the following command:

!pip install pandas
For installation on your system, you can use pip in your command line:

pip install pandas
Installing Pandas using conda:
If you are using the Anaconda distribution of Python, you can use the conda package manager to install pandas. Type the following command in your terminal:

conda install pandas
After installation, you can import and check the installed Pandas version in your Python script as follows:

import pandas as pd

print(pd.__version__)

1.5.3
This will print the version of Pandas installed in your environment to ensure it’s correctly installed.

Pandas Data Structures
Pandas Series
What is a Series? A Series is a one-dimensional labeled array that can hold any data type. It is similar to a column in a spreadsheet or a vector in a mathematical matrix.

Key Features
It can be created from dictionaries, ndarrays, and scalar values.
Each item in a Series object has an index, which is a label that uniquely identifies it.
Series are similar to ndarrays and can be passed into most NumPy functions.
Pandas DataFrame
What is a DataFrame? A DataFrame is a 2-dimensional labeled data structure in Pandas, similar to a table in a relational database, an Excel spreadsheet, or a dictionary of Series objects.

Here is a basic example of a DataFrame:

import pandas as pd

data = {
    'Name': ['John', 'Anna', 'Peter'],
    'Age': [28, 23, 34]
}
df = pd.DataFrame(data)

df

Name	Age
0	John	28
1	Anna	23
2	Peter	34
In this example, ‘Name’ and ‘Age’ are column labels and the numbers 0, 1, 2 on the left are the index labels.

An index in Pandas is a built-in data structure that makes data manipulation and analysis more efficient. It is an immutable array (cannot be changed) and an ordered set. By default Pandas uses a numeric sequence starting from 0 as index.

If you want to customize these index labels, you can do so when creating the DataFrame:

df = pd.DataFrame(data, index=['a', 'b', 'c'])

df

Name	Age
a	John	28
b	Anna	23
c	Peter	34
Now, ‘a’, ‘b’, ‘c’ are the index labels. These can be used with loc method to access specific rows.

Key Features
It can store data of different types (e.g., integer, string, float, Python objects, etc.).
Each row and column has labels for identification.
It is mutable in size, meaning you can modify rows and columns (insert or delete).
Why is it Useful?
Efficient data handling and storage.
Provides numerous built-in methods to manipulate and analyze data.
Integration with plotting libraries for data visualization.
Handles missing data gracefully and flexible reshaping of datasets.
Basic Operations with Series and DataFrames
Creating a Series
You can create a Series from dictionaries, ndarrays, and scalar values.

import pandas as pd

# From a dictionary
s1 = pd.Series({'a': 0, 'b': 1, 'c': 2})
print(s1)

# From an ndarray
s2 = pd.Series(['a', 'b', 'c', 'd'])
print(s2)

# From a scalar
s3 = pd.Series(5, index=[0, 1, 2, 3])
print(s3)

a    0
b    1
c    2
dtype: int64
0    a
1    b
2    c
3    d
dtype: object
0    5
1    5
2    5
3    5
dtype: int64
Indexing a Series
You can access the elements of a Series in a similar way to indexing with native Python data structures

import pandas as pd

s = pd.Series(['a', 'b', 'c', 'd'])

# Accessing a single element using its index
print(s[0])

# Accessing multiple elements using their indices
print(s[[0, 1, 2]])

a
0    a
1    b
2    c
dtype: object
Slicing a Series
Slicing a Series in pandas works in a similar way to slicing in Python.

import pandas as pd

s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
print(s)

# Slice using the index values (explicit index) - Here 'end' value is included
print(s['a':'c'])

# Slice using index numbers (implicit index) - Here 'end' value is excluded
print(s[0:2])

a    1
b    2
c    3
d    4
e    5
dtype: int64
a    1
b    2
c    3
dtype: int64
a    1
b    2
dtype: int64
Remember that when using explicit index (labels ‘a’, ‘b’, ‘c’, etc.), the slice includes the end index - that’s why ‘c’ is included in the result. While when using implicit index, the ‘end’ index is not included in the slice - so the 2nd index (which is ‘c’ and is the third element) is not included.

Creating a DataFrame
You can create a DataFrame from various data types: dictionary, list of lists, list of dictionaries etc. In data engineering, creating a DataFrame is the first step before performing any data manipulation or analysis tasks.

import pandas as pd

# Creating DataFrame
data = {
    'Name': ['John', 'Anna', 'Peter'],
    'Age': [25, 23, 31],
    'Nationality': ["UK", "USA", "UK"]
}
df = pd.DataFrame(data, index=['a', 'b', 'c'])

df

Name	Age	Nationality
a	John	25	UK
b	Anna	23	USA
c	Peter	31	UK
Indexing
Indexing is the process of accessing an element in a sequence using its position in the sequence (its index). In Pandas, indexing refers to accessing rows and columns of data from a DataFrame, whereas slicing refers to accessing a range of rows and columns.

We can access data or range of data from a DataFrame using different methods.

Using column names
Using .loc and .iloc
Using column names:
# Accessing 'Name' column
names = df['Name']
print('- "Name" column:')
print(names)

# Accessing multiple columns
subset = df[['Name', 'Nationality']]
print('- multiple columns:')
print(subset)

- "Name" column:
a     John
b     Anna
c    Peter
Name: Name, dtype: object
- multiple columns:
    Name Nationality
a   John          UK
b   Anna         USA
c  Peter          UK
Using .loc and .iloc:
Pandas provides various methods to index and access data in a DataFrame, including .loc and .iloc. These methods are used to access a group of rows and columns by labels or a boolean array.

.loc

loc is a label-based data selection method which means that we have to pass the name of the row or column which we want to select. This method includes the last element of the range, unlike Python and iloc method.

# Accessing a single row
print('- single row:')
print(df.loc['a'])

# Accessing multiple rows
print('- multiple row:')
print(df.loc[['a', 'b']])

# Accessing rows and specific columns
print('- rows and specific columns:')
print(df.loc[['a', 'b'], 'Name'])

# Accessing all rows and specific columns
print('- all rows and specific columns:')
print(df.loc[:, 'Name'])

- single row:
Name           John
Age              25
Nationality      UK
Name: a, dtype: object
- multiple row:
   Name  Age Nationality
a  John   25          UK
b  Anna   23         USA
- rows and specific columns:
a    John
b    Anna
Name: Name, dtype: object
- all rows and specific columns:
a     John
b     Anna
c    Peter
Name: Name, dtype: object
.iloc

iloc is an integer index-based method which means that we have to pass integer index in the method to select specific rows/columns. This method does not include the last element of the range.

# Accessing first row
print('- first row:')
print(df.iloc[0])

# Accessing first and second rows
print('- first and second rows:')
print(df.iloc[0:2])

# Accessing first row and first column
print('- first row and first column:')
print(df.iloc[0, 0])

# Accessing all rows and first column
print('- all rows and first column:')
print(df.iloc[:, 0])

# Accessing first two rows and first two columns
print('- first two rows and first two columns:')
print(df.iloc[0:2, 0:2])

- first row:
Name           John
Age              25
Nationality      UK
Name: a, dtype: object
- first and second rows:
   Name  Age Nationality
a  John   25          UK
b  Anna   23         USA
- first row and first column:
John
- all rows and first column:
a     John
b     Anna
c    Peter
Name: Name, dtype: object
- first two rows and first two columns:
   Name  Age
a  John   25
b  Anna   23
Slicing
Slicing is used to access a sequence of data in the dataframe. Slicing is often used in data engineering to divide the dataset into smaller chunks for further processing, for example, dividing the data into train and test sets for model training.

# Slice the first three rows
first_three_rows = df[:3]
print(first_three_rows)

# Slice rows from index 1 to 3
subset = df[1:4]
print(subset)

    Name  Age Nationality
a   John   25          UK
b   Anna   23         USA
c  Peter   31          UK
    Name  Age Nationality
b   Anna   23         USA
c  Peter   31          UK
Filtering Data
Filtering in Pandas allows you to select rows that satisfy a certain condition. You can use slicing with boolean indexing for this. Boolean indexing uses a boolean vector to filter the data. The boolean vector is generated by applying a logical condition to the data. The rows corresponding to True in the boolean vector are selected.

# Filter rows where 'Age' is greater than 30
age_above_30 = df[df['Age'] > 30]

print(age_above_30.head())

    Name  Age Nationality
c  Peter   31          UK
You can also combine multiple conditions using the & (and) and | (or) operators.

# Filter rows where 'Age' is greater than 30 and 'Paying_Status' is 'Paid'
age_above_30_and_teacher = df[(df['Age'] > 30) & (df['Nationality'] == 'UK')]
print(age_above_30_and_teacher)

    Name  Age Nationality
c  Peter   31          UK
Remember, when combining conditions, you need to put each condition in parentheses. This is because the & and | operators have higher precedence than the comparison operators like > and ==.

Data Import and Export
Indexing is useful in data engineering to access specific parts of the data for further operations like transformations, calculations etc.

Reading Data From CSV
Reading data from a CSV file is an important task in data engineering and data science. It’s usually the first step when we have to analyze data which is stored in CSV format.

Many programming languages provide libraries to read data from CSV files. For instance, in Python, we use libraries like pandas to read CSV data.

Here is a simple example:

import pandas as pd

# load the data
dataframe = pd.read_csv('https://storage.googleapis.com/rg-ai-bootcamp/pandas/import-data.csv')

# print the first few lines of the dataframe
dataframe.head()

passengerId	class	sex	age	ticket	fare	cabin	embarked
0	1	First	Male	32	A12345	50.0	C10	S
1	2	Second	Female	25	B67890	30.5	E25	C
2	3	Third	Male	18	C24680	10.0	G12	Q
3	4	First	Female	40	D13579	100.0	A5	S
4	5	Second	Male	35	E97531	20.0	B15	S
In this example, the read_csv() function reads a CSV file and converts it into a Pandas DataFrame. dataframe.head() then prints the first 5 rows of the DataFrame.

Considerations while reading CSVs:

Delimiter: A CSV file’s default delimiter is a comma. However, other characters like semicolons can be used. The correct delimiter should be specified when reading a CSV file.

Header: If the first line of the CSV file is a header (names of columns), make sure the library correctly identifies it.

Encoding: CSV files can be written in different encodings. If you encounter a UnicodeDecodeError, you might need to specify the correct encoding.

Writing Data To CSV
We often need to write or export data to a CSV file after manipulating or analyzing it. This written data can then be used in other systems, shared with other teams, or merely stored for future use.

Here is a simple Python example of writing data to a CSV file using pandas:

import pandas as pd

# Create a simple dataframe
new_dataframe = pd.DataFrame({
   'PassengerId': [1, 2, 3],
   'class': ['First', 'Second', 'First'],
   'sex': ['Female', 'Male', 'Male'],
   'age': [28, 24, 35],
   'ticket': ['U64297', 'V91254', 'W72311'],
   'fare': [75.40, 50.00, 100.00],
   'cabin': ['C20', 'A1', 'Z5'],
   'embarked': ['S', 'Q', 'S']
})

# write to a CSV file
new_dataframe.to_csv('./data/export-data.csv', index=False)

# print the dataframe
new_dataframe

PassengerId	class	sex	age	ticket	fare	cabin	embarked
0	1	First	Female	28	U64297	75.4	C20	S
1	2	Second	Male	24	V91254	50.0	A1	Q
2	3	First	Male	35	W72311	100.0	Z5	S
In this case, the to_csv() function writes the DataFrame to a CSV file. The index=False argument prevents pandas from writing row indices into the CSV file.

Considerations while writing CSVs:

Delimiter: While a comma is standard, you might need to use a different character as a delimiter, which can be defined when writing a CSV file.

Header: You can choose to include or exclude the header in your CSV file.

Encoding: You can specify the type of encoding for your CSV file.

These functionalities to read and write CSV files form the basics of handling data in data engineering or data science tasks. They enable you to ingest data from various sources, process it as needed, and store or distribute results in a universally accepted format.

Data Indexing and Selection
Data indexing and selection is a crucial aspect of data manipulation and analysis. It allows us to access specific subsets of data efficiently. This process is particularly important when dealing with large datasets where manual inspection is not feasible.

Selecting Data by Index, Columns, and Labels
In Pandas, a popular data manipulation library in Python, you can select data based on the following 3 things: - Index - Columns - Labels

First, you need to import the necessary libraries and load your data:

import pandas as pd

# Load the data from the .csv file
dataframe = pd.read_csv('https://storage.googleapis.com/rg-ai-bootcamp/pandas/import-data.csv')

dataframe.head(10)

passengerId	class	sex	age	ticket	fare	cabin	embarked
0	1	First	Male	32	A12345	50.00	C10	S
1	2	Second	Female	25	B67890	30.50	E25	C
2	3	Third	Male	18	C24680	10.00	G12	Q
3	4	First	Female	40	D13579	100.00	A5	S
4	5	Second	Male	35	E97531	20.00	B15	S
5	6	Third	Female	28	F86420	15.75	C30	Q
6	7	First	Male	50	G75319	80.50	D8	C
7	8	Second	Female	22	H64208	35.25	E12	S
8	9	Third	Male	19	I35790	8.50	F20	S
9	10	First	Female	45	J86420	90.00	G5	C
Selecting Data by Index
The iloc function is used to select data by its numeric index, similar to how we access elements in a list. The i in iloc stands for integer. We can select a single row, or multiple rows, or even cut a range of rows.

For example, here we want to see in detail the data from the first row, we can access it via index (index = 0)

# Select the first row
first_row = dataframe.iloc[0]

first_row

passengerId         1
class           First
sex              Male
age                32
ticket         A12345
fare             50.0
cabin             C10
embarked            S
Name: 0, dtype: object
Selecting Data by Columns
We can select data by its column name in Pandas using the [] operator. This will return a Series object containing the data in the specified column. In this case, we want to look based on class in the data we have.

# Select the 'embarked' column
class_column = dataframe['class']

class_column.head(10)

0     First
1    Second
2     Third
3     First
4    Second
5     Third
6     First
7    Second
8     Third
9     First
Name: class, dtype: object
Selecting Data by Labels
The loc function in Pandas is used to select rows based on their labels. In a DataFrame, row labels are indexes. By default, DataFrame indexes are numeric, similar to list indexes. However, we can set the index to be any column of the DataFrame, which allows us to use loc to select rows based on the values in that column. After we know the passenger class through the class column, now we want to see more detailed passenger data from the first class.

We can use the following way:

# Set the 'class' column as the index
dataframe.set_index('class', inplace=True)

# Select the row with the label 'First'
first_row = dataframe.loc['First']

first_row

passengerId	sex	age	ticket	fare	cabin	embarked
class							
First	1	Male	32	A12345	50.0	C10	S
First	4	Female	40	D13579	100.0	A5	S
First	7	Male	50	G75319	80.5	D8	C
First	10	Female	45	J86420	90.0	G5	C
First	13	Male	55	M75319	75.5	C8	S
First	16	Female	38	P86420	95.0	F5	S
First	19	Male	60	S75319	70.5	I8	C
Data Manipulation
Adding and Updating Column
Adding Column
You can add a new column to your DataFrame in several ways: 1. Assigning a scalar value to a new column 2. Adding a Series as a new column 3. Adding a calculated column

First, you need to import the necessary libraries and load your data:

import pandas as pd

# Load the data from the .csv file
passenger_df = pd.read_csv('https://storage.googleapis.com/rg-ai-bootcamp/pandas/import-data.csv')

passenger_df.head(10)

passengerId	class	sex	age	ticket	fare	cabin	embarked
0	1	First	Male	32	A12345	50.00	C10	S
1	2	Second	Female	25	B67890	30.50	E25	C
2	3	Third	Male	18	C24680	10.00	G12	Q
3	4	First	Female	40	D13579	100.00	A5	S
4	5	Second	Male	35	E97531	20.00	B15	S
5	6	Third	Female	28	F86420	15.75	C30	Q
6	7	First	Male	50	G75319	80.50	D8	C
7	8	Second	Female	22	H64208	35.25	E12	S
8	9	Third	Male	19	I35790	8.50	F20	S
9	10	First	Female	45	J86420	90.00	G5	C
1. Assigning a scalar value to a new column
This will add a new column to our DataFrame with the supplied scalar values for all the rows. For example, we want to add a column for the discount given to all passengers of 0.15. We can add it in the following way.

# Add a new column in df
passenger_df['discount'] = 0.15

passenger_df.head()

passengerId	class	sex	age	ticket	fare	cabin	embarked	discount
0	1	First	Male	32	A12345	50.0	C10	S	0.15
1	2	Second	Female	25	B67890	30.5	E25	C	0.15
2	3	Third	Male	18	C24680	10.0	G12	Q	0.15
3	4	First	Female	40	D13579	100.0	A5	S	0.15
4	5	Second	Male	35	E97531	20.0	B15	S	0.15
2. Adding a Series as a new column
So what if we want to add the owned status column using Series? We can use the method below. This will add a new column with the values from Series. Series must be the same length as the DataFrame.

# Create a Series
s = pd.Series(['Canceled', 'Active', 'Canceled', 'Active', 'Active'])

# Add the Series as a new column in the DataFrame
passenger_df['status'] = s

passenger_df.head()

passengerId	class	sex	age	ticket	fare	cabin	embarked	discount	status
0	1	First	Male	32	A12345	50.0	C10	S	0.15	Canceled
1	2	Second	Female	25	B67890	30.5	E25	C	0.15	Active
2	3	Third	Male	18	C24680	10.0	G12	Q	0.15	Canceled
3	4	First	Female	40	D13579	100.0	A5	S	0.15	Active
4	5	Second	Male	35	E97531	20.0	B15	S	0.15	Active
3. Adding a column with a calculated value
We can add new columns that are calculated from existing columns. After previously we added discount for all passengers, in this case we want to add a column to see the total fare after the discount.

passenger_df['totalFare'] = passenger_df['fare'] - (passenger_df['fare'] * passenger_df['discount'])

passenger_df.head()

passengerId	class	sex	age	ticket	fare	cabin	embarked	discount	status	totalFare
0	1	First	Male	32	A12345	50.0	C10	S	0.15	Canceled	42.500
1	2	Second	Female	25	B67890	30.5	E25	C	0.15	Active	25.925
2	3	Third	Male	18	C24680	10.0	G12	Q	0.15	Canceled	8.500
3	4	First	Female	40	D13579	100.0	A5	S	0.15	Active	85.000
4	5	Second	Male	35	E97531	20.0	B15	S	0.15	Active	17.000
Updating Columns
Updating a column in a DataFrame is similar to adding a new column. We assign the new values to the column we want to update. The new values could be a scalar, a Series, or a calculated value.

1. Updating a column with a scalar value
This will update the specified column with the supplied scalar value for all rows. For example, we want to change the amount of the discount given.

passenger_df['discount'] = 0.25

passenger_df.head()

passengerId	class	sex	age	ticket	fare	cabin	embarked	discount	status	totalFare
0	1	First	Male	32	A12345	50.0	C10	S	0.25	Canceled	42.500
1	2	Second	Female	25	B67890	30.5	E25	C	0.25	Active	25.925
2	3	Third	Male	18	C24680	10.0	G12	Q	0.25	Canceled	8.500
3	4	First	Female	40	D13579	100.0	A5	S	0.25	Active	85.000
4	5	Second	Male	35	E97531	20.0	B15	S	0.25	Active	17.000
2. Updating a column with a Series
This will update the specified column with the value from Series. Series must be the same length as the DataFrame. In this case, it turns out that we know that the status of some passengers is wrong and we want to replace it using the data series we have.

new_series = pd.Series(['Active', 'Active', 'Active', 'Active', 'Canceled', 'Active', 'Active', 'Canceled','Active', 'Active', 'Canceled', 'Active', 'Active', 'Canceled','Active', 'Canceled', 'Active', 'Active', 'Canceled','Active',])

passenger_df['status'] = new_series

passenger_df.head(10)

passengerId	class	sex	age	ticket	fare	cabin	embarked	discount	status	totalFare
0	1	First	Male	32	A12345	50.00	C10	S	0.25	Active	42.5000
1	2	Second	Female	25	B67890	30.50	E25	C	0.25	Active	25.9250
2	3	Third	Male	18	C24680	10.00	G12	Q	0.25	Active	8.5000
3	4	First	Female	40	D13579	100.00	A5	S	0.25	Active	85.0000
4	5	Second	Male	35	E97531	20.00	B15	S	0.25	Canceled	17.0000
5	6	Third	Female	28	F86420	15.75	C30	Q	0.25	Active	13.3875
6	7	First	Male	50	G75319	80.50	D8	C	0.25	Active	68.4250
7	8	Second	Female	22	H64208	35.25	E12	S	0.25	Canceled	29.9625
8	9	Third	Male	19	I35790	8.50	F20	S	0.25	Active	7.2250
9	10	First	Female	45	J86420	90.00	G5	C	0.25	Active	76.5000
3. Updating a column with a calculated value
We can update columns with values calculated from other columns. For example, after we change the amount of discount given, we also need to change the calculation format and amount.

passenger_df['totalFare'] = passenger_df['fare'] - (passenger_df['fare'] * passenger_df['discount']) 

passenger_df.head(10)

passengerId	class	sex	age	ticket	fare	cabin	embarked	discount	status	totalFare
0	1	First	Male	32	A12345	50.00	C10	S	0.25	Active	37.5000
1	2	Second	Female	25	B67890	30.50	E25	C	0.25	Active	22.8750
2	3	Third	Male	18	C24680	10.00	G12	Q	0.25	Active	7.5000
3	4	First	Female	40	D13579	100.00	A5	S	0.25	Active	75.0000
4	5	Second	Male	35	E97531	20.00	B15	S	0.25	Canceled	15.0000
5	6	Third	Female	28	F86420	15.75	C30	Q	0.25	Active	11.8125
6	7	First	Male	50	G75319	80.50	D8	C	0.25	Active	60.3750
7	8	Second	Female	22	H64208	35.25	E12	S	0.25	Canceled	26.4375
8	9	Third	Male	19	I35790	8.50	F20	S	0.25	Active	6.3750
9	10	First	Female	45	J86420	90.00	G5	C	0.25	Active	67.5000
Updating Column Name
At times, the naming of columns or features may lack consistency, possibly due to variations in letter case, among other factors. Maintaining a uniform naming convention enhances our efficiency when working with these features.

Let’s explore how we can modify or update the names of columns or features in our dataset. For example, we think that the column name sex is taboo or inappropriate, and we want to replace it with gender.

#update the column name
passenger_df = passenger_df.rename(columns = {'sex':'gender'})

passenger_df.head()

passengerId	class	gender	age	ticket	fare	cabin	embarked	discount	status	totalFare
0	1	First	Male	32	A12345	50.0	C10	S	0.25	Active	37.500
1	2	Second	Female	25	B67890	30.5	E25	C	0.25	Active	22.875
2	3	Third	Male	18	C24680	10.0	G12	Q	0.25	Active	7.500
3	4	First	Female	40	D13579	100.0	A5	S	0.25	Active	75.000
4	5	Second	Male	35	E97531	20.0	B15	S	0.25	Canceled	15.000
You can even update multiple column names at a single time. For that, you have to add other column names separated by a comma under the curl braces.

#multiple column update
passenger_df = passenger_df.rename(columns = {'ticket':'ticketNumber','cabin':'cabinNumber'})

passenger_df

passengerId	class	gender	age	ticketNumber	fare	cabinNumber	embarked	discount	status	totalFare
0	1	First	Male	32	A12345	50.00	C10	S	0.25	Active	37.5000
1	2	Second	Female	25	B67890	30.50	E25	C	0.25	Active	22.8750
2	3	Third	Male	18	C24680	10.00	G12	Q	0.25	Active	7.5000
3	4	First	Female	40	D13579	100.00	A5	S	0.25	Active	75.0000
4	5	Second	Male	35	E97531	20.00	B15	S	0.25	Canceled	15.0000
5	6	Third	Female	28	F86420	15.75	C30	Q	0.25	Active	11.8125
6	7	First	Male	50	G75319	80.50	D8	C	0.25	Active	60.3750
7	8	Second	Female	22	H64208	35.25	E12	S	0.25	Canceled	26.4375
8	9	Third	Male	19	I35790	8.50	F20	S	0.25	Active	6.3750
9	10	First	Female	45	J86420	90.00	G5	C	0.25	Active	67.5000
10	11	Second	Male	30	K97531	25.00	A20	S	0.25	Canceled	18.7500
11	12	Third	Female	21	L24680	12.50	B30	Q	0.25	Active	9.3750
12	13	First	Male	55	M75319	75.50	C8	S	0.25	Active	56.6250
13	14	Second	Female	28	N64208	40.25	D12	C	0.25	Canceled	30.1875
14	15	Third	Male	20	O35790	7.50	E20	S	0.25	Active	5.6250
15	16	First	Female	38	P86420	95.00	F5	S	0.25	Canceled	71.2500
16	17	Second	Male	33	Q97531	22.00	G20	S	0.25	Active	16.5000
17	18	Third	Female	26	R24680	13.50	H30	Q	0.25	Active	10.1250
18	19	First	Male	60	S75319	70.50	I8	C	0.25	Canceled	52.8750
19	20	Second	Female	24	T64208	45.25	J12	S	0.25	Active	33.9375
Adding, Updating and Deleting Rows
Adding Rows
There are 2 ways to add rows into Pandas DataFrame object: 1. Use the DataFrame object’s loc attribute. 2. Use the DataFrame object’s concat method.

1. Adding Rows Using loc
What if we want to add the latest passenger data? One of them we can use loc as below. When adding a new row using loc, we can specify the index label of the new row. If this label doesn’t already exist, loc will add a new row with this label to the DataFrame.

new_row = {'passengerId': 21,
           'class':'Third',
           'gender':'Male',
           'age':30,
           'ticketNumber': 'F73925',
           'fare':35,
           'cabinNumber':'A55',
           'embarked': 'C',
           'discount': 0.25,
           'status': 'Active',
           'totalFare': 37.500}

passenger_df.loc[len(passenger_df)] = new_row

passenger_df

passengerId	class	gender	age	ticketNumber	fare	cabinNumber	embarked	discount	status	totalFare
0	1	First	Male	32	A12345	50.00	C10	S	0.25	Active	37.5000
1	2	Second	Female	25	B67890	30.50	E25	C	0.25	Active	22.8750
2	3	Third	Male	18	C24680	10.00	G12	Q	0.25	Active	7.5000
3	4	First	Female	40	D13579	100.00	A5	S	0.25	Active	75.0000
4	5	Second	Male	35	E97531	20.00	B15	S	0.25	Canceled	15.0000
5	6	Third	Female	28	F86420	15.75	C30	Q	0.25	Active	11.8125
6	7	First	Male	50	G75319	80.50	D8	C	0.25	Active	60.3750
7	8	Second	Female	22	H64208	35.25	E12	S	0.25	Canceled	26.4375
8	9	Third	Male	19	I35790	8.50	F20	S	0.25	Active	6.3750
9	10	First	Female	45	J86420	90.00	G5	C	0.25	Active	67.5000
10	11	Second	Male	30	K97531	25.00	A20	S	0.25	Canceled	18.7500
11	12	Third	Female	21	L24680	12.50	B30	Q	0.25	Active	9.3750
12	13	First	Male	55	M75319	75.50	C8	S	0.25	Active	56.6250
13	14	Second	Female	28	N64208	40.25	D12	C	0.25	Canceled	30.1875
14	15	Third	Male	20	O35790	7.50	E20	S	0.25	Active	5.6250
15	16	First	Female	38	P86420	95.00	F5	S	0.25	Canceled	71.2500
16	17	Second	Male	33	Q97531	22.00	G20	S	0.25	Active	16.5000
17	18	Third	Female	26	R24680	13.50	H30	Q	0.25	Active	10.1250
18	19	First	Male	60	S75319	70.50	I8	C	0.25	Canceled	52.8750
19	20	Second	Female	24	T64208	45.25	J12	S	0.25	Active	33.9375
20	21	Third	Male	30	F73925	35.00	A55	C	0.25	Active	37.5000
2. Adding Rows Using concat
Or if we have another DataFrame and want to add it to our existing DataFrame. We can use the concat function. The concat function is used to concat another DataFrame row to the end of the given DataFrame, returning a new DataFrame object.

new_rows = {'passengerId': [22, 23],
            'class':['First','Third'],
            'gender':['Male','Female'],
            'age':[30,30],
            'ticketNumber': ['G76201', 'H43599'],
            'fare':[50,35],
            'cabinNumber':['B5', 'B6'],
            'embarked': ['C', 'S'],
            'discount':[0.25, 0.25],
            'status': ['Active','Active'],
            'totalFare': [37.500, 37.500]}

new_passenger_df = pd.DataFrame(data = new_rows)

passenger_df = pd.concat([passenger_df, new_passenger_df], ignore_index=True)

passenger_df

passengerId	class	gender	age	ticketNumber	fare	cabinNumber	embarked	discount	status	totalFare
0	1	First	Male	32	A12345	50.00	C10	S	0.25	Active	37.5000
1	2	Second	Female	25	B67890	30.50	E25	C	0.25	Active	22.8750
2	3	Third	Male	18	C24680	10.00	G12	Q	0.25	Active	7.5000
3	4	First	Female	40	D13579	100.00	A5	S	0.25	Active	75.0000
4	5	Second	Male	35	E97531	20.00	B15	S	0.25	Canceled	15.0000
5	6	Third	Female	28	F86420	15.75	C30	Q	0.25	Active	11.8125
6	7	First	Male	50	G75319	80.50	D8	C	0.25	Active	60.3750
7	8	Second	Female	22	H64208	35.25	E12	S	0.25	Canceled	26.4375
8	9	Third	Male	19	I35790	8.50	F20	S	0.25	Active	6.3750
9	10	First	Female	45	J86420	90.00	G5	C	0.25	Active	67.5000
10	11	Second	Male	30	K97531	25.00	A20	S	0.25	Canceled	18.7500
11	12	Third	Female	21	L24680	12.50	B30	Q	0.25	Active	9.3750
12	13	First	Male	55	M75319	75.50	C8	S	0.25	Active	56.6250
13	14	Second	Female	28	N64208	40.25	D12	C	0.25	Canceled	30.1875
14	15	Third	Male	20	O35790	7.50	E20	S	0.25	Active	5.6250
15	16	First	Female	38	P86420	95.00	F5	S	0.25	Canceled	71.2500
16	17	Second	Male	33	Q97531	22.00	G20	S	0.25	Active	16.5000
17	18	Third	Female	26	R24680	13.50	H30	Q	0.25	Active	10.1250
18	19	First	Male	60	S75319	70.50	I8	C	0.25	Canceled	52.8750
19	20	Second	Female	24	T64208	45.25	J12	S	0.25	Active	33.9375
20	21	Third	Male	30	F73925	35.00	A55	C	0.25	Active	37.5000
21	22	First	Male	30	G76201	50.00	B5	C	0.25	Active	37.5000
22	23	Third	Female	30	H43599	35.00	B6	S	0.25	Active	37.5000
Updating Rows
You can update the values in a row using the loc indexer. The loc indexer is used to access a group of rows and columns by label(s) or a boolean array.

# Update a row
passenger_df.loc[0, ['passengerId', 'cabinNumber']] = [24, 'C1']
passenger_df.loc[1] = {'passengerId': 2, 'class':'Second', 'gender':'Female', 'age':'18', 'ticketNumber': 'B67890', 'fare':40, 'cabinNumber':'A1','embarked': 'Q', 'discount':0.25, 'status': 'Active', 'totalFare': 22.875}

passenger_df

passengerId	class	gender	age	ticketNumber	fare	cabinNumber	embarked	discount	status	totalFare
0	24	First	Male	32	A12345	50.00	C1	S	0.25	Active	37.5000
1	2	Second	Female	18	B67890	40.00	A1	Q	0.25	Active	22.8750
2	3	Third	Male	18	C24680	10.00	G12	Q	0.25	Active	7.5000
3	4	First	Female	40	D13579	100.00	A5	S	0.25	Active	75.0000
4	5	Second	Male	35	E97531	20.00	B15	S	0.25	Canceled	15.0000
5	6	Third	Female	28	F86420	15.75	C30	Q	0.25	Active	11.8125
6	7	First	Male	50	G75319	80.50	D8	C	0.25	Active	60.3750
7	8	Second	Female	22	H64208	35.25	E12	S	0.25	Canceled	26.4375
8	9	Third	Male	19	I35790	8.50	F20	S	0.25	Active	6.3750
9	10	First	Female	45	J86420	90.00	G5	C	0.25	Active	67.5000
10	11	Second	Male	30	K97531	25.00	A20	S	0.25	Canceled	18.7500
11	12	Third	Female	21	L24680	12.50	B30	Q	0.25	Active	9.3750
12	13	First	Male	55	M75319	75.50	C8	S	0.25	Active	56.6250
13	14	Second	Female	28	N64208	40.25	D12	C	0.25	Canceled	30.1875
14	15	Third	Male	20	O35790	7.50	E20	S	0.25	Active	5.6250
15	16	First	Female	38	P86420	95.00	F5	S	0.25	Canceled	71.2500
16	17	Second	Male	33	Q97531	22.00	G20	S	0.25	Active	16.5000
17	18	Third	Female	26	R24680	13.50	H30	Q	0.25	Active	10.1250
18	19	First	Male	60	S75319	70.50	I8	C	0.25	Canceled	52.8750
19	20	Second	Female	24	T64208	45.25	J12	S	0.25	Active	33.9375
20	21	Third	Male	30	F73925	35.00	A55	C	0.25	Active	37.5000
21	22	First	Male	30	G76201	50.00	B5	C	0.25	Active	37.5000
22	23	Third	Female	30	H43599	35.00	B6	S	0.25	Active	37.5000
Deleting/Dropping Rows
You can delete rows from your DataFrame using the drop() function. This function removes the row or column(s) you specify from your DataFrame. The axis parameter indicates whether you want to drop labels from the index (axis=0 or axis='index') or columns (axis=1 or axis='columns').

# deleting single row
passenger_df = passenger_df.drop(2, axis=0)

# deleting multiple rows
passenger_df = passenger_df.drop([3, 4], axis=0)

passenger_df

passengerId	class	gender	age	ticketNumber	fare	cabinNumber	embarked	discount	status	totalFare
0	24	First	Male	32	A12345	50.00	C1	S	0.25	Active	37.5000
1	2	Second	Female	18	B67890	40.00	A1	Q	0.25	Active	22.8750
5	6	Third	Female	28	F86420	15.75	C30	Q	0.25	Active	11.8125
6	7	First	Male	50	G75319	80.50	D8	C	0.25	Active	60.3750
7	8	Second	Female	22	H64208	35.25	E12	S	0.25	Canceled	26.4375
8	9	Third	Male	19	I35790	8.50	F20	S	0.25	Active	6.3750
9	10	First	Female	45	J86420	90.00	G5	C	0.25	Active	67.5000
10	11	Second	Male	30	K97531	25.00	A20	S	0.25	Canceled	18.7500
11	12	Third	Female	21	L24680	12.50	B30	Q	0.25	Active	9.3750
12	13	First	Male	55	M75319	75.50	C8	S	0.25	Active	56.6250
13	14	Second	Female	28	N64208	40.25	D12	C	0.25	Canceled	30.1875
14	15	Third	Male	20	O35790	7.50	E20	S	0.25	Active	5.6250
15	16	First	Female	38	P86420	95.00	F5	S	0.25	Canceled	71.2500
16	17	Second	Male	33	Q97531	22.00	G20	S	0.25	Active	16.5000
17	18	Third	Female	26	R24680	13.50	H30	Q	0.25	Active	10.1250
18	19	First	Male	60	S75319	70.50	I8	C	0.25	Canceled	52.8750
19	20	Second	Female	24	T64208	45.25	J12	S	0.25	Active	33.9375
20	21	Third	Male	30	F73925	35.00	A55	C	0.25	Active	37.5000
21	22	First	Male	30	G76201	50.00	B5	C	0.25	Active	37.5000
22	23	Third	Female	30	H43599	35.00	B6	S	0.25	Active	37.5000
Sorting Data
Sorting data in a Pandas DataFrame is a common operation that can be done using the sort_values() function.

We can sort by one or more columns. For example, here we want to see the ticket prices paid by passengers by sorting them. Here’s how you do it:

# Sort by 'fare' in ascending order
df_asc = passenger_df.sort_values('fare')

df_asc.head(10)

passengerId	class	gender	age	ticketNumber	fare	cabinNumber	embarked	discount	status	totalFare
14	15	Third	Male	20	O35790	7.50	E20	S	0.25	Active	5.6250
8	9	Third	Male	19	I35790	8.50	F20	S	0.25	Active	6.3750
11	12	Third	Female	21	L24680	12.50	B30	Q	0.25	Active	9.3750
17	18	Third	Female	26	R24680	13.50	H30	Q	0.25	Active	10.1250
5	6	Third	Female	28	F86420	15.75	C30	Q	0.25	Active	11.8125
16	17	Second	Male	33	Q97531	22.00	G20	S	0.25	Active	16.5000
10	11	Second	Male	30	K97531	25.00	A20	S	0.25	Canceled	18.7500
22	23	Third	Female	30	H43599	35.00	B6	S	0.25	Active	37.5000
20	21	Third	Male	30	F73925	35.00	A55	C	0.25	Active	37.5000
7	8	Second	Female	22	H64208	35.25	E12	S	0.25	Canceled	26.4375
# Sort by 'fare' in descending order
df_desc = passenger_df.sort_values('fare', ascending=False)

df_desc.head(10)

passengerId	class	gender	age	ticketNumber	fare	cabinNumber	embarked	discount	status	totalFare
15	16	First	Female	38	P86420	95.00	F5	S	0.25	Canceled	71.2500
9	10	First	Female	45	J86420	90.00	G5	C	0.25	Active	67.5000
6	7	First	Male	50	G75319	80.50	D8	C	0.25	Active	60.3750
12	13	First	Male	55	M75319	75.50	C8	S	0.25	Active	56.6250
18	19	First	Male	60	S75319	70.50	I8	C	0.25	Canceled	52.8750
0	24	First	Male	32	A12345	50.00	C1	S	0.25	Active	37.5000
21	22	First	Male	30	G76201	50.00	B5	C	0.25	Active	37.5000
19	20	Second	Female	24	T64208	45.25	J12	S	0.25	Active	33.9375
13	14	Second	Female	28	N64208	40.25	D12	C	0.25	Canceled	30.1875
1	2	Second	Female	18	B67890	40.00	A1	Q	0.25	Active	22.8750
We can also sort by multiple columns:

# Sort by 'class' and then 'cabinNumber', both in ascending order
df_sorted = passenger_df.sort_values(['class', 'cabinNumber'])

df_sorted.head(10)

passengerId	class	gender	age	ticketNumber	fare	cabinNumber	embarked	discount	status	totalFare
21	22	First	Male	30	G76201	50.00	B5	C	0.25	Active	37.5000
0	24	First	Male	32	A12345	50.00	C1	S	0.25	Active	37.5000
12	13	First	Male	55	M75319	75.50	C8	S	0.25	Active	56.6250
6	7	First	Male	50	G75319	80.50	D8	C	0.25	Active	60.3750
15	16	First	Female	38	P86420	95.00	F5	S	0.25	Canceled	71.2500
9	10	First	Female	45	J86420	90.00	G5	C	0.25	Active	67.5000
18	19	First	Male	60	S75319	70.50	I8	C	0.25	Canceled	52.8750
1	2	Second	Female	18	B67890	40.00	A1	Q	0.25	Active	22.8750
10	11	Second	Male	30	K97531	25.00	A20	S	0.25	Canceled	18.7500
13	14	Second	Female	28	N64208	40.25	D12	C	0.25	Canceled	30.1875
# Sort by 'class' in ascending order, then 'cabinNumber' in descending order
df_sorted = passenger_df.sort_values(['class', 'cabinNumber'], ascending=[True, False])

df_sorted.head(10)

passengerId	class	gender	age	ticketNumber	fare	cabinNumber	embarked	discount	status	totalFare
18	19	First	Male	60	S75319	70.50	I8	C	0.25	Canceled	52.8750
9	10	First	Female	45	J86420	90.00	G5	C	0.25	Active	67.5000
15	16	First	Female	38	P86420	95.00	F5	S	0.25	Canceled	71.2500
6	7	First	Male	50	G75319	80.50	D8	C	0.25	Active	60.3750
12	13	First	Male	55	M75319	75.50	C8	S	0.25	Active	56.6250
0	24	First	Male	32	A12345	50.00	C1	S	0.25	Active	37.5000
21	22	First	Male	30	G76201	50.00	B5	C	0.25	Active	37.5000
19	20	Second	Female	24	T64208	45.25	J12	S	0.25	Active	33.9375
16	17	Second	Male	33	Q97531	22.00	G20	S	0.25	Active	16.5000
7	8	Second	Female	22	H64208	35.25	E12	S	0.25	Canceled	26.4375
After sorting, if you want to reset the index, we can use the reset_index() function:

df_sorted = df_sorted.reset_index(drop=True)

df_sorted

passengerId	class	gender	age	ticketNumber	fare	cabinNumber	embarked	discount	status	totalFare
0	19	First	Male	60	S75319	70.50	I8	C	0.25	Canceled	52.8750
1	10	First	Female	45	J86420	90.00	G5	C	0.25	Active	67.5000
2	16	First	Female	38	P86420	95.00	F5	S	0.25	Canceled	71.2500
3	7	First	Male	50	G75319	80.50	D8	C	0.25	Active	60.3750
4	13	First	Male	55	M75319	75.50	C8	S	0.25	Active	56.6250
5	24	First	Male	32	A12345	50.00	C1	S	0.25	Active	37.5000
6	22	First	Male	30	G76201	50.00	B5	C	0.25	Active	37.5000
7	20	Second	Female	24	T64208	45.25	J12	S	0.25	Active	33.9375
8	17	Second	Male	33	Q97531	22.00	G20	S	0.25	Active	16.5000
9	8	Second	Female	22	H64208	35.25	E12	S	0.25	Canceled	26.4375
10	14	Second	Female	28	N64208	40.25	D12	C	0.25	Canceled	30.1875
11	11	Second	Male	30	K97531	25.00	A20	S	0.25	Canceled	18.7500
12	2	Second	Female	18	B67890	40.00	A1	Q	0.25	Active	22.8750
13	18	Third	Female	26	R24680	13.50	H30	Q	0.25	Active	10.1250
14	9	Third	Male	19	I35790	8.50	F20	S	0.25	Active	6.3750
15	15	Third	Male	20	O35790	7.50	E20	S	0.25	Active	5.6250
16	6	Third	Female	28	F86420	15.75	C30	Q	0.25	Active	11.8125
17	23	Third	Female	30	H43599	35.00	B6	S	0.25	Active	37.5000
18	12	Third	Female	21	L24680	12.50	B30	Q	0.25	Active	9.3750
19	21	Third	Male	30	F73925	35.00	A55	C	0.25	Active	37.5000
Operations on DataFrames
Arithmetic operations
Arithmetic operations can be performed on the numeric columns in DataFrame.

passenger_df['discount'] = passenger_df['discount'] + 0.1
passenger_df['totalFare'] = passenger_df['fare'] - (passenger_df['fare'] * passenger_df['discount']) 
passenger_df

passengerId	class	gender	age	ticketNumber	fare	cabinNumber	embarked	discount	status	totalFare
0	24	First	Male	32	A12345	50.00	C1	S	0.35	Active	32.5000
1	2	Second	Female	18	B67890	40.00	A1	Q	0.35	Active	26.0000
5	6	Third	Female	28	F86420	15.75	C30	Q	0.35	Active	10.2375
6	7	First	Male	50	G75319	80.50	D8	C	0.35	Active	52.3250
7	8	Second	Female	22	H64208	35.25	E12	S	0.35	Canceled	22.9125
8	9	Third	Male	19	I35790	8.50	F20	S	0.35	Active	5.5250
9	10	First	Female	45	J86420	90.00	G5	C	0.35	Active	58.5000
10	11	Second	Male	30	K97531	25.00	A20	S	0.35	Canceled	16.2500
11	12	Third	Female	21	L24680	12.50	B30	Q	0.35	Active	8.1250
12	13	First	Male	55	M75319	75.50	C8	S	0.35	Active	49.0750
13	14	Second	Female	28	N64208	40.25	D12	C	0.35	Canceled	26.1625
14	15	Third	Male	20	O35790	7.50	E20	S	0.35	Active	4.8750
15	16	First	Female	38	P86420	95.00	F5	S	0.35	Canceled	61.7500
16	17	Second	Male	33	Q97531	22.00	G20	S	0.35	Active	14.3000
17	18	Third	Female	26	R24680	13.50	H30	Q	0.35	Active	8.7750
18	19	First	Male	60	S75319	70.50	I8	C	0.35	Canceled	45.8250
19	20	Second	Female	24	T64208	45.25	J12	S	0.35	Active	29.4125
20	21	Third	Male	30	F73925	35.00	A55	C	0.35	Active	22.7500
21	22	First	Male	30	G76201	50.00	B5	C	0.35	Active	32.5000
22	23	Third	Female	30	H43599	35.00	B6	S	0.35	Active	22.7500
Arithmetic operations are essential in feature engineering where you create new features from existing ones to provide more information to your machine learning models.

Aggregation Functions
Aggregation functions in DataFrames are used to perform mathematical computations on groups of data. These functions provide a way to perform operations on groups of values, summarizing the information.

Common aggregation functions include:

count(): Returns the number of non-null values in each DataFrame column.

sum(): Returns the sum of the values for each column.

mean(): Returns the mean of the values for each column.

median(): Returns the median of the values for each column.

min(): Returns the minimum of the values for each column.

max(): Returns the maximum of the values for each column.

std(): Returns the standard deviation of the values for each column.

var(): Returns the variance of values for each column.

first(): Returns the first of the values for each column.

last(): Returns the last of the values for each column.

passenger_df['age'] = passenger_df['age'].astype(int)

# get the mean of 'Age'
mean_age = passenger_df['age'].mean()
print('Mean age: ',mean_age)

# get the standard deviation of 'Age'
std_age = passenger_df['age'].std()
print('std age: ',std_age)

# get the summary statistics of the dataframe
summary = passenger_df.describe()
summary

Mean age:  31.95
std age:  11.966950101711047
passengerId	age	fare	discount	totalFare
count	20.000000	20.00000	20.000000	2.000000e+01	20.000000
mean	14.350000	31.95000	42.350000	3.500000e-01	27.527500
std	6.200806	11.96695	27.327979	1.139065e-16	17.763186
min	2.000000	18.00000	7.500000	3.500000e-01	4.875000
25%	9.750000	23.50000	20.437500	3.500000e-01	13.284375
50%	14.500000	30.00000	37.625000	3.500000e-01	24.456250
75%	19.250000	34.25000	55.125000	3.500000e-01	35.831250
max	24.000000	60.00000	95.000000	3.500000e-01	61.750000
This is often the first step in any data analysis process to understand the distribution and central tendencies of the data.

Applying Functions in DataFrames
Applying functions is a way to perform operations on a DataFrame or a series (column of a DataFrame). This is done using the apply() function in Pandas. The apply() function takes a function as an argument and applies this function to an entire DataFrame or series.

DataFrame.apply()
When used on a DataFrame, apply() applies the function to each column of the DataFrame (default behavior), returning a series where each element is the result of applying the function to a column. If you want to apply the function to each row, you can set the axis parameter to 1.

For example, here we want to make a recent fare adjustment by increasing the fare by 5 dollars. Then, we also need to make adjustments to the totalFare. We can do it in the following way.

# Add passenger 'fare' by adding 5 to the 'fare'
passenger_df['fare'] = passenger_df['fare'].apply(lambda x: x + 5)

# Define the function to be applied
def calculate_total_fare(row):
    return row['fare'] - (row['fare'] * row['discount'])

# Use the function on every row in the dataframe
passenger_df['totalFare'] = passenger_df.apply(calculate_total_fare, axis=1)

passenger_df

passengerId	class	gender	age	ticketNumber	fare	cabinNumber	embarked	discount	status	totalFare
0	24	First	Male	32	A12345	55.00	C1	S	0.35	Active	35.7500
1	2	Second	Female	18	B67890	45.00	A1	Q	0.35	Active	29.2500
5	6	Third	Female	28	F86420	20.75	C30	Q	0.35	Active	13.4875
6	7	First	Male	50	G75319	85.50	D8	C	0.35	Active	55.5750
7	8	Second	Female	22	H64208	40.25	E12	S	0.35	Canceled	26.1625
8	9	Third	Male	19	I35790	13.50	F20	S	0.35	Active	8.7750
9	10	First	Female	45	J86420	95.00	G5	C	0.35	Active	61.7500
10	11	Second	Male	30	K97531	30.00	A20	S	0.35	Canceled	19.5000
11	12	Third	Female	21	L24680	17.50	B30	Q	0.35	Active	11.3750
12	13	First	Male	55	M75319	80.50	C8	S	0.35	Active	52.3250
13	14	Second	Female	28	N64208	45.25	D12	C	0.35	Canceled	29.4125
14	15	Third	Male	20	O35790	12.50	E20	S	0.35	Active	8.1250
15	16	First	Female	38	P86420	100.00	F5	S	0.35	Canceled	65.0000
16	17	Second	Male	33	Q97531	27.00	G20	S	0.35	Active	17.5500
17	18	Third	Female	26	R24680	18.50	H30	Q	0.35	Active	12.0250
18	19	First	Male	60	S75319	75.50	I8	C	0.35	Canceled	49.0750
19	20	Second	Female	24	T64208	50.25	J12	S	0.35	Active	32.6625
20	21	Third	Male	30	F73925	40.00	A55	C	0.35	Active	26.0000
21	22	First	Male	30	G76201	55.00	B5	C	0.35	Active	35.7500
22	23	Third	Female	30	H43599	40.00	B6	S	0.35	Active	26.0000
Introduction to Data Cleaning
What is Data Cleaning?
Data cleaning, also known as data cleansing or data scrubbing, is the process of identifying and correcting or removing errors, inaccuracies, and inconsistencies in datasets.

This process is crucial in improving the quality and reliability of data, which is particularly important in data analysis and machine learning models where the output quality is directly dependent on the input quality.

In the context of the Python programming language, the Pandas library is often used for data cleaning. Pandas is a powerful data manipulation tool that provides functions for reading, writing, and modifying datasets.

Why is Data Cleaning Important?
Data cleaning is a critical step in the data analysis process for several reasons:

Improving Data Quality: Raw data often contains errors, outliers, or inconsistencies that can distort analysis results. Data cleaning helps to ensure that the data used for analysis is accurate and reliable.

Enhancing Data Accuracy: Inaccurate data can lead to inaccurate conclusions. By cleaning data, you can ensure that your analyses and models are based on the most accurate information possible.

Boosting Efficiency: Clean data is easier to work with and can make data analysis processes more efficient.

Better Decision Making: Clean data leads to more reliable analysis, which in turn leads to better decision-making. This is particularly important in fields like business or research where decisions need to be data-driven.

Ensuring Compliance: In some industries, maintaining clean data is a regulatory requirement. Data cleaning can help ensure compliance with these regulations.

Data Cleaning Process
data-cleaning-in-data-science.png (Sumber: knowledgehut)

Handling Missing Data in Pandas
Missing data is a common issue in data analysis. It refers to the absence of data in a column of a dataset. In Python’s Pandas library, missing data is represented by NaN (Not a Number) or None.

Detecting Missing Data
Pandas provides the isnull() and notnull() functions to detect missing data. These functions return a Boolean mask indicating whether each element in the DataFrame is missing or not.

Let’s take a look at the customer data we used earlier. We will find out if there is missing data in the customer data that we have.

import pandas as pd

airbnb_df = pd.read_csv('https://storage.googleapis.com/rg-ai-bootcamp/pandas/airbnb-data.csv')
airbnb_df

id	name	host_id	host_identity_verified	host_name	neighbourhood_group	neighbourhood	country	country_code	instant_bookable	room_type	construction_year	price	service_fee	minimum_nights	number_of_reviews	review_rate_number	calculated_host_listings_count	license
0	1001254.0	Clean & quiet apt home by the park	8.001449e+10	unconfirmed	Madaline	Brooklyn	Kensington	United States	US	False	Private room	2020.0	$966	$193	10.0	9.0	4.0	6.0	NaN
1	1002102.0	Skylit Midtown Castle	5.233517e+10	verified	Jenna	Manhattan	Midtown	United States	US	False	Entire home/apt	2007.0	$142	$28	30.0	45.0	4.0	2.0	NaN
2	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
3	1002403.0	THE VILLAGE OF HARLEM....NEW YORK !	7.882924e+10	NaN	Elise	Manhattan	Harlem	United States	US	True	Private room	2005.0	$620	$124	3.0	0.0	5.0	1.0	NaN
4	1002755.0	NaN	8.509833e+10	unconfirmed	Garry	Brooklyn	Clinton Hill	United States	US	True	Entire home/apt	2005.0	$368	$74	30.0	270.0	4.0	1.0	NaN
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
3631	2998453.0	Upper West Side elegance. Riverside	3.204065e+10	unconfirmed	Poppi	Manhattan	Upper West Side	United States	US	False	Entire home/apt	2008.0	$620	$124	1.0	5.0	1.0	2.0	NaN
3632	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
3633	2999005.0	BIG, BRIGHT, STYLISH + CONVENIENT	4.074627e+10	unconfirmed	Kerstin	Brooklyn	Bedford-Stuyvesant	United States	US	True	Private room	2020.0	$674	$135	3.0	90.0	2.0	1.0	NaN
3634	2999005.0	BIG, BRIGHT, STYLISH + CONVENIENT	4.074627e+10	unconfirmed	Kerstin	Brooklyn	Bedford-Stuyvesant	United States	US	True	Private room	2020.0	$674	$135	3.0	90.0	2.0	1.0	NaN
3635	2999557.0	Exclusive Upper East Side Studio	2.446375e+10	verified	Ellen	Manhattan	Upper East Side	United States	US	True	Entire home/apt	2022.0	$655	$131	1.0	0.0	5.0	1.0	NaN
3636 rows × 19 columns

# Detect missing values
airbnb_df.isnull()

id	name	host_id	host_identity_verified	host_name	neighbourhood_group	neighbourhood	country	country_code	instant_bookable	room_type	construction_year	price	service_fee	minimum_nights	number_of_reviews	review_rate_number	calculated_host_listings_count	license
0	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	True
1	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	True
2	True	True	True	True	True	True	True	True	True	True	True	True	True	True	True	True	True	True	True
3	False	False	False	True	False	False	False	False	False	False	False	False	False	False	False	False	False	False	True
4	False	True	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	True
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
3631	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	True
3632	True	True	True	True	True	True	True	True	True	True	True	True	True	True	True	True	True	True	True
3633	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	True
3634	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	True
3635	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	True
3636 rows × 19 columns

This will print a DataFrame with the same size as the original, but with True in place where the original DataFrame had NaN or None, and False elsewhere. as can be seen that the row with index 2 has a value of true in each column, which means that there is missing data in that row.

To be clear, if we want to know the amount and percentage of missing data, we can do it in the following way:

# Calculate the total amount of missing data
total = airbnb_df.isnull().sum().sort_values(ascending=False)

# Calculates the percentage of missing data
persentase = (airbnb_df.isnull().sum()/airbnb_df.isnull().count()*100).sort_values(ascending=False)

# Creates a new DataFrame to display the results
missing_data = pd.concat([total, persentase], axis=1, keys=['Total', 'Persentase'])

missing_data

Total	Persentase
license	3636	100.000000
construction_year	138	3.795380
review_rate_number	96	2.640264
minimum_nights	87	2.392739
country_code	82	2.255226
instant_bookable	82	2.255226
host_identity_verified	78	2.145215
name	60	1.650165
country	53	1.457646
neighbourhood_group	31	0.852585
host_name	23	0.632563
neighbourhood	19	0.522552
service_fee	18	0.495050
calculated_host_listings_count	18	0.495050
price	14	0.385039
number_of_reviews	10	0.275028
room_type	3	0.082508
host_id	3	0.082508
id	3	0.082508
Dropping Missing Data
Pandas provides the dropna() function to remove missing data. By default, dropna() removes any row containing at least one missing value.

However, dropna() is more flexible than this. It provides several parameters that allow you to control how missing values are dropped:

Axis: By default, dropna() drops rows (axis=0). But you can also make it drop columns by setting axis=1. py     # Drop columns with missing values     df_dropped = df.dropna(axis=1) This will drop any column that contains at least one missing value.

How: By default, dropna() drops a row or column if it contains any missing values (how='any'). But you can also make it drop only rows or columns where all values are missing by setting how='all'.

# Drop rows where all values are missing
df_dropped = df.dropna(how='all')

This will drop any row where all values are missing.

Subset: You can specify a subset of columns to consider when dropping rows.

# Drop rows where 'house_rules' is missing
df_dropped = df.dropna(subset=['house_rules'])

This will drop any row where ‘house_rules’ is missing.

Inplace: By default, dropna() returns a new DataFrame and leaves the original unchanged. If you want to modify the original DataFrame, you can set inplace=True.

# Drop rows with missing values in the original DataFrame
df.dropna(inplace=True)

This will drop rows with missing values directly in the original DataFrame.

As seen earlier the license column and some rows have no data at all, we will try to remove them using dropna().

# Drop the column 'license' which has no data at all
airbnb_df = airbnb_df.drop(['license'], axis=1)

# Drop any line where 'id', 'name', 'host_id' and 'host_name' are missing
airbnb_df = airbnb_df.dropna(subset=['id', 'name', 'host_id', 'host_name'])

# Drop rows where all values are missing
airbnb_df = airbnb_df.dropna(how='all')

airbnb_df

id	name	host_id	host_identity_verified	host_name	neighbourhood_group	neighbourhood	country	country_code	instant_bookable	room_type	construction_year	price	service_fee	minimum_nights	number_of_reviews	review_rate_number	calculated_host_listings_count
0	1001254.0	Clean & quiet apt home by the park	8.001449e+10	unconfirmed	Madaline	Brooklyn	Kensington	United States	US	False	Private room	2020.0	$966	$193	10.0	9.0	4.0	6.0
1	1002102.0	Skylit Midtown Castle	5.233517e+10	verified	Jenna	Manhattan	Midtown	United States	US	False	Entire home/apt	2007.0	$142	$28	30.0	45.0	4.0	2.0
3	1002403.0	THE VILLAGE OF HARLEM....NEW YORK !	7.882924e+10	NaN	Elise	Manhattan	Harlem	United States	US	True	Private room	2005.0	$620	$124	3.0	0.0	5.0	1.0
5	1003689.0	Entire Apt: Spacious Studio/Loft by central park	9.203760e+10	verified	Lyndon	Manhattan	East Harlem	United States	US	False	Entire home/apt	2009.0	$204	$41	10.0	9.0	3.0	1.0
6	1004098.0	Large Cozy 1 BR Apartment In Midtown East	4.549855e+10	verified	Michelle	Manhattan	Murray Hill	United States	US	True	Entire home/apt	2013.0	$577	$115	3.0	74.0	3.0	1.0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
3630	2997901.0	Brooklyn NY, Comfy,Spacious Repose!	5.626794e+10	verified	Benjamin	Brooklyn	Bushwick	United States	US	False	Private room	2007.0	$767	$153	3.0	40.0	2.0	1.0
3631	2998453.0	Upper West Side elegance. Riverside	3.204065e+10	unconfirmed	Poppi	Manhattan	Upper West Side	United States	US	False	Entire home/apt	2008.0	$620	$124	1.0	5.0	1.0	2.0
3633	2999005.0	BIG, BRIGHT, STYLISH + CONVENIENT	4.074627e+10	unconfirmed	Kerstin	Brooklyn	Bedford-Stuyvesant	United States	US	True	Private room	2020.0	$674	$135	3.0	90.0	2.0	1.0
3634	2999005.0	BIG, BRIGHT, STYLISH + CONVENIENT	4.074627e+10	unconfirmed	Kerstin	Brooklyn	Bedford-Stuyvesant	United States	US	True	Private room	2020.0	$674	$135	3.0	90.0	2.0	1.0
3635	2999557.0	Exclusive Upper East Side Studio	2.446375e+10	verified	Ellen	Manhattan	Upper East Side	United States	US	True	Entire home/apt	2022.0	$655	$131	1.0	0.0	5.0	1.0
3562 rows × 18 columns

This will remove the licence column and row where all values are missing as well as rows from the id, name, host_id and host_name columns which have missing data.

Filling Missing Data in Pandas
Instead of dropping missing values, we can also fill them with some value using the fillna() function. Pandas provides the fillna() function to fill missing values in a DataFrame. This function is highly flexible and allows you to fill missing values in a variety of ways.

However, fillna() provides several parameters that allow you to control how missing values are filled:

Value: This is the value that will replace missing values. It can be a constant, or a dictionary, Series, or DataFrame that specifies different fill values for different columns.

# Fill missing values with different values for different columns
df_filled = df.fillna({'construction_year': 'unknown', 'minimum_nights': df['minimum_nights'].mean()})

Method: This specifies the method to use to fill missing values. Options include ‘forward fill’ (ffill or pad), which fills missing values with the previous value in the column, and ‘backward fill’ (bfill or backfill), which fills missing values with the next value in the column.

# Forward fill
df_filled = df.fillna(method='ffill')

# Backward fill
df_filled = df.fillna(method='bfill')

Axis: By default, fillna() fills missing values along the rows (axis=0). But you can also make it fill along the columns by setting axis=1.

# Forward fill along columns
df_filled = df.fillna(method='ffill', axis=1)

Inplace: By default, fillna() returns a new DataFrame and leaves the original unchanged. If you want to modify the original DataFrame, you can set inplace=True.

# Fill missing values in the original DataFrame
df.fillna("Unknown", inplace=True)

As seen earlier, some of the columns have missing values. Now we will try to fill in the missing values using fillna().

airbnb_df = airbnb_df.fillna({
'construction_year': 'unknown',
'review_rate_number': airbnb_df['review_rate_number'].mean(),
'minimum_nights': airbnb_df['minimum_nights'].mean(),
'instant_bookable': 'unknown',
'country_code': 'unknown',
'host_identity_verified': 'unknown',
'country': 'unknown',
'neighbourhood_group': 'unknown',
'neighbourhood': 'unknown',
'service_fee': 'unknown',
'calculated_host_listings_count': airbnb_df['calculated_host_listings_count'].mean(),
'price': 'unknown',
'number_of_reviews': airbnb_df['number_of_reviews'].mean(),
})

airbnb_df

id	name	host_id	host_identity_verified	host_name	neighbourhood_group	neighbourhood	country	country_code	instant_bookable	room_type	construction_year	price	service_fee	minimum_nights	number_of_reviews	review_rate_number	calculated_host_listings_count
0	1001254.0	Clean & quiet apt home by the park	8.001449e+10	unconfirmed	Madaline	Brooklyn	Kensington	United States	US	False	Private room	2020.0	$966	$193	10.0	9.0	4.0	6.0
1	1002102.0	Skylit Midtown Castle	5.233517e+10	verified	Jenna	Manhattan	Midtown	United States	US	False	Entire home/apt	2007.0	$142	$28	30.0	45.0	4.0	2.0
3	1002403.0	THE VILLAGE OF HARLEM....NEW YORK !	7.882924e+10	unknown	Elise	Manhattan	Harlem	United States	US	True	Private room	2005.0	$620	$124	3.0	0.0	5.0	1.0
5	1003689.0	Entire Apt: Spacious Studio/Loft by central park	9.203760e+10	verified	Lyndon	Manhattan	East Harlem	United States	US	False	Entire home/apt	2009.0	$204	$41	10.0	9.0	3.0	1.0
6	1004098.0	Large Cozy 1 BR Apartment In Midtown East	4.549855e+10	verified	Michelle	Manhattan	Murray Hill	United States	US	True	Entire home/apt	2013.0	$577	$115	3.0	74.0	3.0	1.0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
3630	2997901.0	Brooklyn NY, Comfy,Spacious Repose!	5.626794e+10	verified	Benjamin	Brooklyn	Bushwick	United States	US	False	Private room	2007.0	$767	$153	3.0	40.0	2.0	1.0
3631	2998453.0	Upper West Side elegance. Riverside	3.204065e+10	unconfirmed	Poppi	Manhattan	Upper West Side	United States	US	False	Entire home/apt	2008.0	$620	$124	1.0	5.0	1.0	2.0
3633	2999005.0	BIG, BRIGHT, STYLISH + CONVENIENT	4.074627e+10	unconfirmed	Kerstin	Brooklyn	Bedford-Stuyvesant	United States	US	True	Private room	2020.0	$674	$135	3.0	90.0	2.0	1.0
3634	2999005.0	BIG, BRIGHT, STYLISH + CONVENIENT	4.074627e+10	unconfirmed	Kerstin	Brooklyn	Bedford-Stuyvesant	United States	US	True	Private room	2020.0	$674	$135	3.0	90.0	2.0	1.0
3635	2999557.0	Exclusive Upper East Side Studio	2.446375e+10	verified	Ellen	Manhattan	Upper East Side	United States	US	True	Entire home/apt	2022.0	$655	$131	1.0	0.0	5.0	1.0
3562 rows × 18 columns

This will fill in missing values in the review_rate_number , minimum_nights, calculated_host_listings_count and number_of_reviews columns with the average and missing values in the other columns with unknown.

Data Type Conversion in Pandas
Data type conversion, also known as type casting, is an important step in data cleaning. It involves converting data from one type to another. This is often necessary because the type of data that Pandas infers upon loading a dataset might not always be what you want.

Converting Data Types in Pandas
Data type conversion is a crucial step in data cleaning, especially when the data is imported from various sources which may categorize data in different types. Pandas provides the astype() function to convert data types.

As can be seen, the data in the id, host_id, minimum_nights, number_of_reviews,review_rate_number, and calculated_host_listings_count columns are of the float data type. We will convert the data into an integer using astype() function.

# Change the column with the value `float` to `int`
airbnb_df[['id', 'host_id', 'minimum_nights', 'number_of_reviews','review_rate_number','calculated_host_listings_count']] = airbnb_df[['id', 'host_id', 'minimum_nights', 'number_of_reviews','review_rate_number','calculated_host_listings_count']].astype(int)

airbnb_df

id	name	host_id	host_identity_verified	host_name	neighbourhood_group	neighbourhood	country	country_code	instant_bookable	room_type	construction_year	price	service_fee	minimum_nights	number_of_reviews	review_rate_number	calculated_host_listings_count
0	1001254	Clean & quiet apt home by the park	80014485718	unconfirmed	Madaline	Brooklyn	Kensington	United States	US	False	Private room	2020.0	$966	$193	10	9	4	6
1	1002102	Skylit Midtown Castle	52335172823	verified	Jenna	Manhattan	Midtown	United States	US	False	Entire home/apt	2007.0	$142	$28	30	45	4	2
3	1002403	THE VILLAGE OF HARLEM....NEW YORK !	78829239556	unknown	Elise	Manhattan	Harlem	United States	US	True	Private room	2005.0	$620	$124	3	0	5	1
5	1003689	Entire Apt: Spacious Studio/Loft by central park	92037596077	verified	Lyndon	Manhattan	East Harlem	United States	US	False	Entire home/apt	2009.0	$204	$41	10	9	3	1
6	1004098	Large Cozy 1 BR Apartment In Midtown East	45498551794	verified	Michelle	Manhattan	Murray Hill	United States	US	True	Entire home/apt	2013.0	$577	$115	3	74	3	1
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
3630	2997901	Brooklyn NY, Comfy,Spacious Repose!	56267937797	verified	Benjamin	Brooklyn	Bushwick	United States	US	False	Private room	2007.0	$767	$153	3	40	2	1
3631	2998453	Upper West Side elegance. Riverside	32040648122	unconfirmed	Poppi	Manhattan	Upper West Side	United States	US	False	Entire home/apt	2008.0	$620	$124	1	5	1	2
3633	2999005	BIG, BRIGHT, STYLISH + CONVENIENT	40746270692	unconfirmed	Kerstin	Brooklyn	Bedford-Stuyvesant	United States	US	True	Private room	2020.0	$674	$135	3	90	2	1
3634	2999005	BIG, BRIGHT, STYLISH + CONVENIENT	40746270692	unconfirmed	Kerstin	Brooklyn	Bedford-Stuyvesant	United States	US	True	Private room	2020.0	$674	$135	3	90	2	1
3635	2999557	Exclusive Upper East Side Studio	24463750542	verified	Ellen	Manhattan	Upper East Side	United States	US	True	Entire home/apt	2022.0	$655	$131	1	0	5	1
3562 rows × 18 columns

Removing Duplicates in Pandas
Duplicate data can occur in your DataFrame for a variety of reasons, and it’s often necessary to remove these duplicates in order to perform accurate analysis.

Identifying Duplicates
Pandas provides the duplicated() function to identify duplicate rows. This function returns a Boolean Series that is True for each row that is a duplicate of a previous row and False otherwise.

Now we will find out if there are duplicate data in the customer data that we have.

# Identify duplicate rows
airbnb_df.duplicated()

0       False
1       False
3       False
5       False
6       False
        ...  
3630     True
3631    False
3633    False
3634     True
3635    False
Length: 3562, dtype: bool
By default, duplicated() considers all columns. If you look at the results, there is some duplicate data.

Removing Duplicates
Pandas provides the drop_duplicates() function to remove duplicate rows. This function returns a new DataFrame where duplicate rows have been removed. By default, drop_duplicates() considers all columns and keeps the first occurrence of each duplicate. If you want to consider only certain columns when dropping duplicates, or if you want to keep the last occurrence instead, you can pass arguments.

# Remove duplicate rows in the original DataFrame
airbnb_df.drop_duplicates(inplace=True)

airbnb_df.tail()

id	name	host_id	host_identity_verified	host_name	neighbourhood_group	neighbourhood	country	country_code	instant_bookable	room_type	construction_year	price	service_fee	minimum_nights	number_of_reviews	review_rate_number	calculated_host_listings_count
3628	2997348	East Village Studio	29175111219	unconfirmed	Sheila	Manhattan	East Village	United States	US	True	Entire home/apt	2010.0	$297	$59	4	216	1	1
3629	2997901	Brooklyn NY, Comfy,Spacious Repose!	56267937797	verified	Benjamin	Brooklyn	Bushwick	United States	US	False	Private room	2007.0	$767	$153	3	40	2	1
3631	2998453	Upper West Side elegance. Riverside	32040648122	unconfirmed	Poppi	Manhattan	Upper West Side	United States	US	False	Entire home/apt	2008.0	$620	$124	1	5	1	2
3633	2999005	BIG, BRIGHT, STYLISH + CONVENIENT	40746270692	unconfirmed	Kerstin	Brooklyn	Bedford-Stuyvesant	United States	US	True	Private room	2020.0	$674	$135	3	90	2	1
3635	2999557	Exclusive Upper East Side Studio	24463750542	verified	Ellen	Manhattan	Upper East Side	United States	US	True	Entire home/apt	2022.0	$655	$131	1	0	5	1
Now, duplicate data has been removed. Remember, it’s important to understand your data and your specific use case before removing duplicates. Sometimes, what appears to be a duplicate might actually be valid data.

Outlier Detection and Treatment in Pandas
Outliers are data points that are significantly different from other observations. They can be caused by variability in the data or experimental errors. Outliers can skew statistical measures and data distributions, leading to misleading results.

Detecting Outliers
There are several ways to detect outliers. A common method is to use the IQR (interquartile range) rule. The IQR is the range between the first quartile (25th percentile) and the third quartile (75th percentile) of the data.

Any data point that falls below the first quartile minus 1.5 times the IQR or above the third quartile plus 1.5 times the IQR is considered an outlier.

In this case we want to find out which hosts have the best value, but there are differences in data in the number_of_reviews column, therefore we need to identify which data are outliers.

Here’s how to detect outliers in the number_of_reviews column using IQR rules:

Q1 = airbnb_df['number_of_reviews'].quantile(0.25)
Q3 = airbnb_df['number_of_reviews'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print(f"lower bound: {lower_bound}")
print(f"upper bound: {upper_bound}")

lower bound: -113.5
upper bound: 210.5
After we determine the lower and upper bound, we will identify which data are outliers in our dataframe.

# Identify outliers
outliers = airbnb_df[(airbnb_df['number_of_reviews'] < lower_bound) | (airbnb_df['number_of_reviews'] > upper_bound)]

outliers.head()

id	name	host_id	host_identity_verified	host_name	neighbourhood_group	neighbourhood	country	country_code	instant_bookable	room_type	construction_year	price	service_fee	minimum_nights	number_of_reviews	review_rate_number	calculated_host_listings_count
10	1005754	Large Furnished Room Near B'way	79384379533	verified	Evelyn	Manhattan	Hell's Kitchen	United States	US	True	Private room	2005.0	$1,018	$204	2	430	3	1
21	1011277	Chelsea Perfect	73862528370	verified	Alberta	Manhattan	Chelsea	United States	unknown	unknown	Private room	2008.0	$460	unknown	1	260	3	1
34	1018457	front room/double bed	69410526955	unconfirmed	Byron	Manhattan	Harlem	United States	unknown	unknown	Private room	2004.0	$770	$154	3	242	3	3
37	1020114	back room/bunk beds	25066620900	verified	Alfred	Manhattan	Harlem	United States	unknown	unknown	Private room	2021.0	$545	$109	3	273	3	3
41	1022323	Cute apt in artist's home	88653822946	verified	Joyce	Brooklyn	Bushwick	United States	US	True	Entire home/apt	2005.0	$1,097	$219	2	231	3	2
In this example, outliers is a DataFrame that contains all rows in airbnb_df where number_of_reviews is an outlier.

Based on the previous results, data with id 1005754 is outlier data because it has number_of_reviews with a number above upper_bound (upper_bound = 210.5) of 430.

airbnb_df.set_index('id', inplace=True)
outlier_data = airbnb_df.loc[1005754]

outlier_data

name                              Large Furnished Room Near B'way
host_id                                               79384379533
host_identity_verified                                   verified
host_name                                                  Evelyn
neighbourhood_group                                     Manhattan
neighbourhood                                      Hell's Kitchen
country                                             United States
country_code                                                   US
instant_bookable                                             True
room_type                                            Private room
construction_year                                          2005.0
price                                                     $1,018 
service_fee                                                 $204 
minimum_nights                                                  2
number_of_reviews                                             430
review_rate_number                                              3
calculated_host_listings_count                                  1
Name: 1005754, dtype: object
Treating Outliers
Once you’ve detected outliers, you need to decide how to handle them. There are several strategies for this: - Removing outliers: If you’re confident that the outliers are due to errors, you might choose to remove them. - Capping outliers: Instead of removing outliers, you might choose to cap them at the lower and upper bounds. - Imputing outliers: Another strategy is to replace outliers with some imputed value, like the mean or median.

Removing Outliers
If you’re confident that the outliers in your data are due to errors, you might choose to remove them. This can be done using boolean indexing to create a new DataFrame that only includes rows where the value is not an outlier.

# Remove outliers
df_no_outliers = airbnb_df[(airbnb_df['number_of_reviews'] >= lower_bound) & (airbnb_df['number_of_reviews'] <= upper_bound)]

df_no_outliers.head()

name	host_id	host_identity_verified	host_name	neighbourhood_group	neighbourhood	country	country_code	instant_bookable	room_type	construction_year	price	service_fee	minimum_nights	number_of_reviews	review_rate_number	calculated_host_listings_count
id																	
1001254	Clean & quiet apt home by the park	80014485718	unconfirmed	Madaline	Brooklyn	Kensington	United States	US	False	Private room	2020.0	$966	$193	10	9	4	6
1002102	Skylit Midtown Castle	52335172823	verified	Jenna	Manhattan	Midtown	United States	US	False	Entire home/apt	2007.0	$142	$28	30	45	4	2
1002403	THE VILLAGE OF HARLEM....NEW YORK !	78829239556	unknown	Elise	Manhattan	Harlem	United States	US	True	Private room	2005.0	$620	$124	3	0	5	1
1003689	Entire Apt: Spacious Studio/Loft by central park	92037596077	verified	Lyndon	Manhattan	East Harlem	United States	US	False	Entire home/apt	2009.0	$204	$41	10	9	3	1
1004098	Large Cozy 1 BR Apartment In Midtown East	45498551794	verified	Michelle	Manhattan	Murray Hill	United States	US	True	Entire home/apt	2013.0	$577	$115	3	74	3	1
In this example, df_no_outliers is a new DataFrame where all outliers in the number_of_reviews column have been removed.

Capping Outliers
Instead of removing outliers, you might choose to cap them at the lower and upper bounds. This can be done using the clip() function, which caps values below a lower threshold at the lower threshold and values above an upper threshold at the upper threshold.

# Cap outliers
df_capped = airbnb_df.copy()
df_capped['number_of_reviews'] = df_capped['number_of_reviews'].clip(lower_bound, upper_bound)

outlier_data_capped = df_capped.loc[1005754]

outlier_data_capped

name                              Large Furnished Room Near B'way
host_id                                               79384379533
host_identity_verified                                   verified
host_name                                                  Evelyn
neighbourhood_group                                     Manhattan
neighbourhood                                      Hell's Kitchen
country                                             United States
country_code                                                   US
instant_bookable                                             True
room_type                                            Private room
construction_year                                          2005.0
price                                                     $1,018 
service_fee                                                 $204 
minimum_nights                                                  2
number_of_reviews                                           210.5
review_rate_number                                              3
calculated_host_listings_count                                  1
Name: 1005754, dtype: object
In this example, df_capped is a new DataFrame with all outliers in the number_of_reviews column bounded at the bottom and top bounds. If you pay attention, the data with id 1005754 now has a value of number_of_reviews which has been adjusted to upper_bound of 210.5.

Imputing Outliers
Another strategy is to replace outliers with some imputed value, like the mean, median, or mode. This can be done using boolean indexing to replace outlier values.

# Impute outliers with the median
df_imputed = airbnb_df.copy()
df_imputed.loc[(df_imputed['number_of_reviews'] < lower_bound) | (df_imputed['number_of_reviews'] > upper_bound), 'number_of_reviews'] = df_imputed['number_of_reviews'].median()

outlier_data_imputed = df_imputed.loc[1005754]

outlier_data_imputed

name                              Large Furnished Room Near B'way
host_id                                               79384379533
host_identity_verified                                   verified
host_name                                                  Evelyn
neighbourhood_group                                     Manhattan
neighbourhood                                      Hell's Kitchen
country                                             United States
country_code                                                   US
instant_bookable                                             True
room_type                                            Private room
construction_year                                          2005.0
price                                                     $1,018 
service_fee                                                 $204 
minimum_nights                                                  2
number_of_reviews                                              30
review_rate_number                                              3
calculated_host_listings_count                                  1
Name: 1005754, dtype: object
In this example, df_imputed is a new DataFrame where all outliers in the number_of_reviews column have been replaced with the median.

Exercise Pandas
!pip install rggrader

# @title #### Student Identity
student_id = "your student id" # @param {type:"string"}
name = "your name" # @param {type:"string"}

# @title #### 00. Filtering Data
from rggrader import submit
import pandas as pd

students = {
    'Name': ['Eric', 'Clay', 'Edward', 'Paul', 'Tara', 'Cris'],
    'Math': [60, 76, 90, 55, 69, 88],
    'Economy': [77, 83, 66, 71, 88, 91]
}
students_df = pd.DataFrame(students)

# TODO: Filter students data where 'Math' and 'Economy' score are greater than 75
# Put your code here:
filtered_students = none


# ---- End of your code ----

# Submit Method
assignment_id = "00_pandas"
question_id = "00_filtering-data"
submit(student_id, name, assignment_id, filtered_students.to_string(), question_id)

# Expected Output:
#    Name  Math  Economy
# 1  Clay    76       83
# 5  Cris    88       91

# @title #### 01. Replace Characters in string
from rggrader import submit
import pandas as pd

cars = {
    'Name': ['NSX', 'Supra', 'WRZ', 'Jesko', 'Veyron', 'P1'],
    'Power': ['500HP', '512HP', '510HP', '700HP', '800HP', '600HP'],
}
cars_df = pd.DataFrame(cars)

# TODO: Replace 'HP' character in 'Power' column
# Put your code here:


# ---- End of your code ----

# Submit Method
assignment_id = "00_pandas"
question_id = "01_replace-characters-in-string"
submit(student_id, name, assignment_id, cars_df.to_string(), question_id)

# Expected Output:
#      Name Power
# 0     NSX   500
# 1   Supra   512
# 2     WRZ   510
# 3   Jesko   700
# 4  Veyron   800
# 5      P1   600

# @title #### 03. Fill missing value
# from rggrader import submit
import numpy as np
import pandas as pd

athletes = {
    'Name': ['Eric', 'Clay', 'Edward', 'Paul', 'Tara', 'Cris'],
    'Medals': [7, 4, np.NaN, 5, 8, np.NaN],
}
athletes_df = pd.DataFrame(athletes)

# TODO: Fill the missing data in the 'Medals' column using the average of values
# Put your code here: 


# ---- End of your code ----

# Submit Method
assignment_id = "00_pandas"
question_id = "02_fill-missing-value"
submit(student_id, name, assignment_id, athletes_df.to_string(), question_id)

# Expected Output:
#      Name  Medals
# 0    Eric     7.0
# 1    Clay     4.0
# 2  Edward     6.0
# 3    Paul     5.0
# 4    Tara     8.0
# 5    Cris     6.0