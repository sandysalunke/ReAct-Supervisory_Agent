import sqlite3
import random
from datetime import datetime, timedelta

conn = sqlite3.connect("sales_multi.db")
cursor = conn.cursor()

# --- Create Tables ---

cursor.execute("""
CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT,
    city TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT,
    category TEXT,
    price REAL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    total_amount REAL,
    order_date TEXT,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY(product_id) REFERENCES products(product_id)
)
""")

# --- Insert Customers ---
cities = ["Mumbai", "Pune", "Delhi", "Bangalore"]

for i in range(1, 201):
    cursor.execute(
        "INSERT INTO customers VALUES (?, ?, ?)",
        (i, f"Customer_{i}", random.choice(cities))
    )

# --- Insert Products ---
categories = ["Electronics", "Clothing", "Food"]

for i in range(1, 51):
    price = round(random.uniform(10, 1000), 2)
    cursor.execute(
        "INSERT INTO products VALUES (?, ?, ?, ?)",
        (i, f"Product_{i}", random.choice(categories), price)
    )

# --- Insert Orders ---
start_date = datetime(2023, 1, 1)

for i in range(1, 3000):
    customer_id = random.randint(1, 200)
    product_id = random.randint(1, 50)
    quantity = random.randint(1, 5)

    # get product price
    cursor.execute("SELECT price FROM products WHERE product_id=?", (product_id,))
    price = cursor.fetchone()[0]

    total_amount = round(price * quantity, 2)
    date = start_date + timedelta(days=random.randint(0, 365))

    cursor.execute("""
        INSERT INTO orders (customer_id, product_id, quantity, total_amount, order_date)
        VALUES (?, ?, ?, ?, ?)
    """, (customer_id, product_id, quantity, total_amount, date.strftime('%Y-%m-%d')))

conn.commit()
conn.close()

print("✅ Multi-table database created: sales_multi.db")