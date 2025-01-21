import os
import grpc
from concurrent import futures
import time
import psycopg2
import vehicle_data_pb2
import vehicle_data_pb2_grpc


def get_connection():
    dbname = os.getenv("DBNAME")
    user = os.getenv("USER")
    password = os.getenv("PASSWORD")
    host = os.getenv("HOST")
    port = os.getenv("PORT")
    return psycopg2.connect(
        database=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )

class ServiceVehicle(vehicle_data_pb2_grpc.ServiceVehicleServicer):
    def GetVehicleData(self, request, context):
        start_time = request.start_time
        end_time = request.end_time

        with get_connection() as conn:
            with conn.cursor() as cursor:
                min_time_query = "SELECT MIN(detection_time) AS min_time FROM vehicle_count"
                cursor.execute(min_time_query)
                min_time_result = cursor.fetchone()
                min_time = min_time_result[0]

                if min_time is None:
                    return vehicle_data_pb2.VehicleData(
                        car_number_left=0,
                        car_number_right=0
                    )

                if start_time < min_time:
                    query = f"""
                        SELECT
                        MAX(car_number_left) - 0 AS car_number_left_diff,
                        MAX(car_number_right) - 0 AS car_number_right_diff
                        FROM vehicle_count 
                        WHERE detection_time BETWEEN '{start_time}' AND '{end_time}'
                    """
                else:
                    query = f"""
                        SELECT
                        MAX(car_number_left) - MIN(car_number_left) AS car_number_left_diff,
                        MAX(car_number_right) - MIN(car_number_right) AS ar_number_right_diff
                        FROM vehicle_count 
                        WHERE detection_time BETWEEN '{start_time}' AND '{end_time}'
                    """

                cursor.execute(query)
                result = cursor.fetchone()

        if result is None:
            return vehicle_data_pb2.VehicleData(
                car_number_left=0,
                car_number_right=0
            )
        else:
            return vehicle_data_pb2.VehicleData(
                car_number_left=result[0],
                car_number_right=result[1]
            )



def serve():
    ip_address = os.getenv("GRPC_HOST")
    port = os.getenv("GRPC_PORT")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    vehicle_data_pb2_grpc.add_ServiceVehicleServicer_to_server(ServiceVehicle(), server)
    server.add_insecure_port(f'{ip_address}:{port}')
    server.start()
    print(f"Сервер запущен на {ip_address}:{port}")
    try:
        while True:
            time.sleep(20)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
