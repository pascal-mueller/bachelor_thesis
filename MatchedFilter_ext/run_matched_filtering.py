import csv
import os

segments_file = open('segments.csv')
segments = csv.reader(segments_file)

header = next(segments)

total = 0
month = 30*24*60*60

for i, start, end in segments:
    print(i)
    start = int(start)
    end = int(end)

    flag = total + (end - start) > month
    
    if total + (end - start) > month:
        end = start + month - total

    times = list(range(start, end, 10240))

    
    # If last diff is smaller than 512, replace end.
    if len(times) > 0 and end - times[-1] < 1024:
        times[-1] = end
    else:
        times.append(end)
    
    if len(times) == 0:
        times = [end]

    for i in range(len(times)-1):
        # Check if job was already successfully run
        filename = f"outputa/{times[i] - 80}-{times[i+1] + 16}-out.hdf"
        try:
            f = h5py.File(filename, 'r')
            print(f"{filename} was run successfully already.")
            f.close()
        except:
            if os.path.isfile(filename):
                continue
                #os.remove(filename)

            cmd = f"bsub -n 16 -W 02:00 ./run_search.sh {times[i] - 80} {times[i+1] + 16}"
            print(cmd)
            #os.system(cmd)


    total += end - start
    if flag == True:
        break


print(total, month)
