
import sqlite3

def readData(readQuery):
    # Connect to DB
    conn = sqlite3.connect("sales.db?mode=ro", uri=True)
    cursor = conn.cursor()

    # Run query
    cursor.execute(readQuery)

    # Fetch data
    rows = cursor.fetchall()

    # Close connection
    conn.close()

    return rows