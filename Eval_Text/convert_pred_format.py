import os 

det_path = "D:\\Text\\ctw1500-test-pred"

with open('pred_file_ori.txt', 'w') as w:
    list_files = sorted(os.listdir(det_path))
    total_str = ''
    for i, file in enumerate(list_files):
        file_str = ''
        with open(os.path.join(det_path, file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line.split(' ')
                # data.insert(0, 'x y w h')
                # data.insert(0, 0.98)
                # data.insert(0, str(i))
                # print(data)
                new_str = ' '.join(str(item) for item in data) 
                file_str += new_str 
        total_str += file_str 
    print(total_str)

    w.write(total_str)



    

