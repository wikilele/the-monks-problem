
class DataSet:
    def __init__(self,filepath):
        self.filepath = filepath

    def encode_1ofk(self,inputs):
        max_values = [3,3,2,3,4,2]
        encoded_inputs = []
        for i in range(len(inputs)):
            for j in range(max_values[i]):
                if j + 1 == inputs[i]:
                    encoded_inputs.append(1)
                else:
                    encoded_inputs.append(0)
        return encoded_inputs


    def get_set(self):
        data_set = []
        with open(self.filepath, 'r') as fp:
            for line in fp:
                words_list = line.strip().split()
                output_class = int(words_list[0])
                inputs = list(map(int,words_list[1 : 7]))
                inputs = self.encode_1ofk(inputs)
                data_set.append((inputs, [output_class]))


        return data_set
