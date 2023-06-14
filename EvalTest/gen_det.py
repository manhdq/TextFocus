import os 
annopath = 'annotations'
detpath = 'detections'

list_files = os.listdir(annopath)

for file in list_files:
    out_save = os.path.join(detpath, file)
    with open(os.path.join(annopath, file), 'r') as f:
        data = f.readlines()
        change_data = []
        for line in data: 
            line = line.strip()
            line = line.split(' ')
            remain = line[:5]
            change = [int(s) for s in line[5:]]
            after_change = []
            for i in range(len(change)):
                after_change.append(change[i] + 8)
            content = ' '.join(remain) + ' ' + ' '.join(map(str, after_change))
            change_data.append(content)

    with open(out_save, 'w', encoding='utf-8') as f:
        for line in change_data:
            f.write(line + '\n')
            
