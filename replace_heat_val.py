with open('temp.txt','w') as file2:
    with open('pan12.gene2freq.txt','r') as file:
        for line in file:
            x = line.split()[0]
            y = line.split()[1]  
            if(line.split()[1]=='0'):
                y = '0.1'
            file2.write(x+'\t'+y+'\n')
    #filedata = file.read()
