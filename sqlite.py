import sqlite3 

connection = sqlite3.connect("e3_student.db")


cursor = connection.cursor()

table_info = """
create table STUDENT (NAME VARCHAR(25), CLASS VARCHAR(25),GRADE VARCHAR(5),MARKS INT)
"""

cursor.execute(table_info)

cursor.execute(''' Insert Into STUDENT values('Malik','Generative AI','B',47)''')
cursor.execute(''' Insert Into STUDENT values('Ali','Data Science','A',1)''')
cursor.execute(''' Insert Into STUDENT values('Hassan','Math','B',87)''')
cursor.execute(''' Insert Into STUDENT values('Hussain','Generative AI','B',77)''')
cursor.execute(''' Insert Into STUDENT values('Mustafa','Data Science','AA',97)''')
cursor.execute(''' Insert Into STUDENT values('Qasim','Generative AI','B',7)''')
cursor.execute(''' Insert Into STUDENT values('Hamza','Data Science','B',27)''')
cursor.execute(''' Insert Into STUDENT values('Atif','Generative AI','A',77)''')
cursor.execute(''' Insert Into STUDENT values('Samad','Math','C',77)''')

print("The inserted records are")

data = cursor.execute('''Select * from STUDENT''')

for row in data:
    print(row)

connection.commit()

connection.close()