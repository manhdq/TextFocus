
import os 

gt_path = "D:\\Text\\ctw1500-test-gt"

with open('gt_file_ori.txt', 'w') as w:
    list_files = sorted(os.listdir(gt_path))
    total_str = ''
    for i, file in enumerate(list_files):
        file_str = ''
        with open(os.path.join(gt_path, file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line.split(',')[:-1]
                data.insert(0, 'x y w h')
                data.insert(0, 0)
                data.insert(0, str(i))
                print(data)

                new_str = ' '.join(str(item) for item in data) 
                file_str += new_str + '\n'
        total_str += file_str 
    print(total_str)

    w.write(total_str)



    

