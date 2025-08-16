import serial 
import csv 
import time

#pip install pyserial

def read_serial_data(port, baud_rate, num_readings, output_file):
    try:
        #open serial connection
        with serial.Serial(port, baud_rate, timeout=1) as ser:
            print(f"Connected to {port} at {baud_rate} baud rate")
            
            #open csv file for writing
            with open(output_file, mode='w', newline='')as file:
                writer = csv.writer(file)
                writer.writerow(["Reading Number", "Date"]) #Write Head
                
                time.sleep(2) #DElay 2 sec bet reading
                # REad specified number of data series
                for i in range(num_readings):
                    data= ser.readline().decode('utf-8').strip() #Read
                    
                    if data:
                        print(f"reading{i+1}: {data}")
                        writer.writerow([i+1, data]) #Write data to CSV
                    else:
                        print(f"Reading {i+1}: NO data received")
                        writer.writerow([i+1, "No data received"])
                        
                        time.sleep(2) 
                        
    except serial.SerialException as e:
         print(f"Error:{e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")
if __name__ == "__main__" :
    #REplace "Com5" with your deivce port and set the appropriate baud
    port="com5" #Example: '/dev/ttyUSB0' for linux or 'COM3' for Windows
    baud_rate= 115200 #Adjust as per your device's configuration
    num_readings=25
    output_file = 'serial_data.csv'  # file to save t
    
    read_serial_data(port, baud_rate, num_readings, output_file)