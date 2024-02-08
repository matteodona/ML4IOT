import redis
from datetime import datetime, timedelta
import time
import psutil
import uuid
import argparse as ap



class RedisClient():
    def __init__(self,host,port,username,password):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.redis_client = redis.Redis(host = self.host, port = self.port, username = self.username, password = self.password)
        self.mac_address = hex(uuid.getnode())


    def is_connected(self):
        is_connected = self.redis_client.ping()
        print(f'Redis Connected: {is_connected} \n')


    def create_key(self, key_name, retention_time_ms = None):
        try:
            self.redis_client.ts().create(key_name, retention_msecs=retention_time_ms)
        except redis.ResponseError:
            pass

    def ts_plugged_seconds(self):
            """
            This function create a new timeseries mac_address:plugged_seconds that, every hour, 
            automatically stores how many seconds the power has been plugged in the last hour.
            """
            try:
                self.redis_client.ts().create(f'{self.mac_address}:plugged_seconds', retention_msecs=30 * 24 * 60 * 60 * 1000)
                self.redis_client.ts().createrule(f'{self.mac_address}:power', f'{self.mac_address}:plugged_seconds', 'sum', bucket_size_msec=60 * 60 * 1000)
            except redis.ResponseError:
                pass


    def add_ts_battery_data(self, duration=None):
        """
        parameters : duration in seconds

        This function will acquire data about percentage of battery level and power plugged for the specified duration and store those data on the redis timeseries.
        If no value is provided, the acquisition will continue without a fixed limit
        The function prints the execution time
        """
        

        start_time = time.time()

        iterations = 1

        while True:

            timestamp = time.time()
            timestamp_ms = int(timestamp * 1000)
            battery_level = psutil.sensors_battery().percent
            power_plugged = int(psutil.sensors_battery().power_plugged)
        
            self.redis_client.ts().add(f'{self.mac_address}:battery', timestamp_ms, battery_level)
            self.redis_client.ts().add(f'{self.mac_address}:power', timestamp_ms, power_plugged)

            #Print some log information
            if (iterations % 50 == 0):
                print(f'Data added : {iterations}')
            iterations += 1

            # Check if the desired duration has passed
            if duration is not None:
                if time.time() - start_time >= duration:
                    break
            
            time.sleep(1)


        






if __name__ == '__main__' :
    #Defining parser
    parser = ap.ArgumentParser()
    parser.add_argument('--duration', type = int, default = None, help = 'Time (s) to add data to timeseries')
    parser.add_argument('--host', type=str)
    parser.add_argument('--port', type=str)
    parser.add_argument('--user', type=str)
    parser.add_argument('--password', type=str)
    args = parser.parse_args()

    #Create Redis client and check the connection to the database
    redis_client = RedisClient(args.host,args.port,args.user,args.password)
    redis_client.is_connected()


    mac_address = redis_client.mac_address


    #Create keys 
    print("Creating battery and power keys... \n")
    redis_client.create_key(f'{mac_address}:battery', retention_time_ms=24 * 60 * 60 * 1000)
    redis_client.create_key(f'{mac_address}:power', retention_time_ms=24 * 60 * 60 * 1000)
    
    #Create keys for plugged seconds and adding plugging data
    print("Creating plugged seconds keys... \n")
    redis_client.ts_plugged_seconds() 
 

    #Add battery data
    print("Adding data.. \n")
    redis_client.add_ts_battery_data(duration=args.duration)
    print("Finish adding data \n")
#

    print('Process ended.')
