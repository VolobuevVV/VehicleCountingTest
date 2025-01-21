import os
import time

import psycopg2

class DB:
    def __init__(self):
        self.dbname = os.getenv("DBNAME")
        self.user = os.getenv("USER")
        self.password = os.getenv("PASSWORD")
        self.host = os.getenv("HOST")
        self.port = os.getenv("PORT")
        self.create_table()

    def get_connection(self):
        return psycopg2.connect(
            database=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        )
    def create_table(self):
        connected = False
        attempts = 0
        while not connected:
            try:
                with self.get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute('''
                                            CREATE TABLE IF NOT EXISTS vehicle_count (
                                                car_number_left INT,
                                                car_number_right INT,
                                                detection_time INT
                                            ); 
                                            ''')
                        conn.commit()
                        print("Подключение к TimescaleDB прошло успешно!")
                        connected = True
            except Exception as e:
                attempts += 1
                print(f"Попытка {attempts}: Подключение к TimescaleDB не удалось. Ошибка: {e}")
                time.sleep(10)
    def run(self, output_queue ):

        data = output_queue.get(timeout=60)
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('''INSERT INTO vehicle_count (car_number_left, car_number_right, detection_time) VALUES (%s, %s, %s)''', data)
                    conn.commit()
                    print(f"Данные вставлены: left={data[0]}, right={data[1]}, time={data[2]}")
        except Exception as e:
            print(f"Ошибка при вставке данных: {e}")

